from __future__ import annotations
import json
from typing import Optional
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
            pass  # already exists

    def exists(self, path: str) -> bool:
        blob = self.container.get_blob_client(path)
        return blob.exists()

    def upload_bytes(self, path: str, data: bytes, content_type: str):
        blob = self.container.get_blob_client(path)
        cs = ContentSettings(content_type=content_type)
        blob.upload_blob(data, overwrite=True, content_settings=cs)

    def upload_json(self, path: str, obj: dict):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.upload_bytes(path, data, "application/json")

    def download_json(self, path: str) -> dict:
        blob = self.container.get_blob_client(path)
        data = blob.download_blob().readall()
        return json.loads(data)

    def list_prefix(self, prefix: str):
        return self.container.list_blobs(name_starts_with=prefix)
