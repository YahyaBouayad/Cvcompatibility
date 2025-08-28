import json
from typing import Dict
from storage import BlobStore

class StateStore:
    """
    Petit store JSON dans le Blob : state/<name>.json
    """
    def __init__(self, store: BlobStore):
        self.store = store

    def _path(self, name: str) -> str:
        return f"state/{name}.json"

    def load(self, name: str) -> Dict:
        p = self._path(name)
        if not self.store.exists(p):
            return {}
        try:
            return self.store.download_json(p)
        except Exception:
            return {}

    def save(self, name: str, obj: Dict):
        p = self._path(name)
        self.store.upload_json(p, obj)
