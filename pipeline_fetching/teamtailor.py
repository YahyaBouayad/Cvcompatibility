from __future__ import annotations
from typing import Dict, Iterable, Optional, Any
import requests
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception
from rate_limit import TokenBucket
import config

class TTError(Exception):
    pass

def _is_retryable(e: Exception):
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
        self.bucket = TokenBucket(rate_per_sec or config.RATE_MAX_CALLS_PER_SEC)

    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(6),
        retry=retry_if_exception(_is_retryable),
        reraise=True
    )
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None):
        self.bucket.acquire(1)
        url = self.base_url + path.lstrip("/")
        r = self.session.get(url, headers=self.headers, params=params, timeout=30)
        if r.status_code == 429:
            # laisser Tenacity remonter une exception retryable :
            raise TTError("429 Too Many Requests")
        if r.status_code >= 400:
            raise TTError(f"HTTP {r.status_code}: {r.text}")
        return r.json()

    def list_resource(self, resource: str, per_page: int = 100, include=None, extra_params=None):
        params = {"per_page": per_page}
        if include:
            # JSON:API attend une liste séparée par des virgules, *avec* des tirets
            params["include"] = ",".join(include)
        if extra_params:
            params.update(extra_params or {})

        page = 1
        while True:
            params["page"] = page
            self.bucket.acquire(1)
            r = self.session.get(self.base_url + resource, headers=self.headers, params=params, timeout=30)
            if r.status_code == 400:
                # aide au debug en cas d'include invalide
                raise RuntimeError(f"[400] Invalid include for {resource}: {r.text}")
            r.raise_for_status()
            data = r.json()
            for item in data.get("data", []):
                yield item
            if not data.get("links", {}).get("next"):
                break
            page += 1


    def get_entity(self, resource: str, entity_id: str | int, include: Optional[list[str]] = None) -> Dict[str, Any]:
        params = {}
        if include:
            params["include"] = ",".join(include)
        return self._get(f"{resource}/{entity_id}", params=params)

    def get_related(self, resource: str, entity_id: str | int, relationship: str) -> Dict[str, Any]:
        return self._get(f"{resource}/{entity_id}/{relationship}")

    def list_uploads_by_candidate(self, candidate_id: str | int, per_page: int = 100):
        # Fallback pratique: filtre côté /uploads si besoin
        return self.list_resource("uploads", per_page=per_page, extra_params={"filter[candidate]": candidate_id})
