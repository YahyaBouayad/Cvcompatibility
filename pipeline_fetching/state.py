from __future__ import annotations
from typing import Dict, Any
from storage import BlobStore

class StateStore:
    def __init__(self, store: BlobStore):
        self.store = store

    def _path(self, resource: str) -> str:
        return f"tt/state/{resource}.json"

    def load(self, resource: str) -> Dict[str, Any]:
        path = self._path(resource)
        try:
            return self.store.download_json(path)
        except Exception:
            return {"last_run_ts": None, "items": 0, "last_updated_at": None}

    def save(self, resource: str, state: Dict[str, Any]):
        self.store.upload_json(self._path(resource), state)
