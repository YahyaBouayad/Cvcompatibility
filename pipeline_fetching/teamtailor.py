from __future__ import annotations

from typing import Dict, Iterable, Optional, Any, List, Union, Tuple
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception

from rate_limit import TokenBucket
import config


class TTError(Exception):
    pass


def _is_retryable(e: Exception) -> bool:
    return isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout, TTError))


class TTClient:
    def __init__(self, headers: Optional[Dict[str, str]] = None, rate_per_sec: float = None):
        self.base_url = config.TEAMTAILOR_BASE_URL.rstrip("/") + "/"
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Token token={config.TEAMTAILOR_API_KEY}",
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/vnd.api+json",
            "X-Api-Version": config.TEAMTAILOR_API_VERSION,
        }
        if headers:
            self.headers.update(headers)

        pool = int(config.MAX_WORKERS) * int(config.HTTP_POOL_MULTIPLIER)
        adapter = HTTPAdapter(
            pool_connections=pool,
            pool_maxsize=pool,
            max_retries=Retry(total=0, redirect=0, connect=0, read=0, status=0, backoff_factor=0),
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers["Connection"] = "keep-alive"

        self.bucket = TokenBucket(rate_per_sec or config.RATE_MAX_CALLS_PER_SEC)

    # -------- HTTP core --------
    @retry(wait=wait_exponential_jitter(initial=1, max=30),
           stop=stop_after_attempt(6),
           retry=retry_if_exception(_is_retryable),
           reraise=True)
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.bucket.acquire(1)
        url = self.base_url + path.lstrip("/")
        r = self.session.get(url, headers=self.headers, params=params, timeout=30)
        if r.status_code == 429:
            raise TTError("429 Too Many Requests")
        if r.status_code >= 400:
            raise TTError(f"HTTP {r.status_code}: {r.text}")
        return r.json()

    def _safe_request_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        attempt = 0
        backoff = 0.5
        while True:
            self.bucket.acquire(1)
            r = self.session.get(url, headers=self.headers, params=params, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                attempt += 1
                if attempt >= 6:
                    raise TTError(f"HTTP {r.status_code} after retries: {r.text[:180]}")
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 8.0)
                continue
            if r.status_code >= 400:
                raise TTError(f"HTTP {r.status_code}: {r.text[:180]}")
            return r.json()

    # -------- JSON:API utils --------
    def list_resource(self, resource: str, per_page: int = 100,
                      include: Optional[List[str]] = None,
                      extra_params: Optional[Dict[str, Any]] = None,
                      max_pages: int = None) -> Iterable[Dict[str, Any]]:
        params: Dict[str, Any] = {"per_page": per_page}
        if include:
            params["include"] = ",".join(include)
        if extra_params:
            params.update(extra_params or {})
        page = 1
        url = self.base_url + resource
        guard = max_pages or int(config.MAX_PAGES)
        while True:
            params["page"] = page
            data = self._safe_request_json(url, params)
            for it in data.get("data", []):
                yield it
            if not data.get("links", {}).get("next"):
                break
            page += 1
            if page > guard:
                raise TTError(f"Pagination overflow on {resource}: exceeded {guard} pages")

    def get_entity(self, resource: str, entity_id: Union[str, int],
                   include: Optional[List[str]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if include:
            params["include"] = ",".join(include)
        return self._get(f"{resource}/{entity_id}", params=params)

    def get_related(self, resource: str, entity_id: Union[str, int], relationship: str) -> Dict[str, Any]:
        return self._get(f"{resource}/{entity_id}/{relationship}")

    # ================== RÉSUMÉS via ATTRIBUTS DU CANDIDAT ==================

    def get_candidate_resume_links(self, candidate_id: Union[str, int]) -> Dict[str, Optional[str]]:
        """
        Un seul appel: GET /candidates/{id}
        Retourne les URLs éventuelles pour 'resume' et 'original-resume'.
        """
        resp = self.get_entity("candidates", candidate_id)
        data = resp.get("data") or {}
        attrs = data.get("attributes") or {}
        # Variantes possibles selon tenant
        resume_url = attrs.get("resume") or attrs.get("resume-url") or attrs.get("resume_url")
        original_url = attrs.get("original-resume") or attrs.get("original_resume") or attrs.get("original-resume-url")
        return {"resume": resume_url, "original": original_url}

    @retry(wait=wait_exponential_jitter(initial=1, max=30),
           stop=stop_after_attempt(6),
           retry=retry_if_exception(_is_retryable),
           reraise=True)
    def download_content(self, url: str) -> Tuple[bytes, str]:
        """
        Télécharge un binaire en renvoyant (content, content_type).
        Pas d'headers API (URL signée publique).
        """
        r = self.session.get(url, timeout=60)
        if r.status_code >= 400:
            raise TTError(f"Download failed ({r.status_code}): {r.text[:180]}")
        content_type = r.headers.get("Content-Type", "application/octet-stream")
        return r.content, content_type
