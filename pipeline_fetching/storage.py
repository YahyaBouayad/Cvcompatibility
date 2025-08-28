from __future__ import annotations
import json
from typing import Optional, Dict, Any
from azure.storage.blob import BlobServiceClient, ContentSettings
import config

class BlobStore:
    def __init__(self, conn_str: Optional[str] = None, container: Optional[str] = None):
        self.conn_str = conn_str or config.AZURE_STORAGE_CONNECTION_STRING
        self.container_name = container or config.AZURE_BLOB_CONTAINER
        self.client = BlobServiceClient.from_connection_string(self.conn_str)
        self.container = self.client.get_container_client(self.container_name)
        try:
            self.container.create_container()
        except Exception:
            pass

    def exists(self, path: str) -> bool:
        try:
            self.container.get_blob_client(path).get_blob_properties()
            return True
        except Exception:
            return False

    def get_blob_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        try:
            props = self.container.get_blob_client(path).get_blob_properties()
            return props.metadata or {}
        except Exception:
            return None

    def upload_bytes(self, path: str, data: bytes, content_type: str = "application/octet-stream",
                     metadata: Optional[Dict[str, str]] = None, overwrite: bool = True):
        blob = self.container.get_blob_client(path)
        cs = ContentSettings(content_type=content_type, cache_control="no-cache")
        blob.upload_blob(
            data,
            overwrite=overwrite,
            content_settings=cs,
            metadata=metadata or {},
            max_concurrency=4,  # upload streaming multi-part, évite un gros PUT bloquant
        )

    def upload_json(self, path: str, obj: dict, metadata: Optional[Dict[str, str]] = None):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.upload_bytes(path, data, "application/json", metadata=metadata)

    def download_json(self, path: str) -> dict:
        blob = self.container.get_blob_client(path)
        data = blob.download_blob().readall()
        return json.loads(data)

    def list_prefix(self, prefix: str):
        return self.container.list_blobs(name_starts_with=prefix)
