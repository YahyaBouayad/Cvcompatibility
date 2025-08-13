from __future__ import annotations
from typing import Dict, Iterable, Optional, Any
import requests
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception
from .rate_limit import TokenBucket
from . import config

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
    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.bucket.acquire()
        url = self.base_url + endpoint.lstrip("/")
        resp = self.session.get(url, headers=self.headers, params=params, timeout=30)
        if resp.status_code == 429:
            raise TTError(f"429 rate limit: {resp.text}")
        if resp.status_code >= 500:
            raise TTError(f"{resp.status_code}: {resp.text}")
        if not resp.ok:
            raise TTError(f"{resp.status_code}: {resp.text}")
        return resp.json()

    def list_resource(
        self,
        resource: str,
        per_page: int = 100,
        status: Optional[str] = None,
        include: Optional[list[str]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Iterable[Dict[str, Any]]:
        page = 1
        params: Dict[str, Any] = {"per_page": per_page, "page": page}
        if status:
            params["status[]"] = status
            params["status"] = status
        if include:
            params["include"] = ",".join(include)
        if extra_params:
            params.update(extra_params)
        while True:
            params["page"] = page
            data = self._get(resource, params=params)
            items = data.get("data", [])
            if not items:
                break
            for item in items:
                yield item
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
