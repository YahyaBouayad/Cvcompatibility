from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from .teamtailor import TTClient
from .storage import BlobStore
from .state import StateStore
from .mime_utils import extension_from_mime
import requests
import logging

logger = logging.getLogger("cvcompat.enrich")

def enrich_entity(client: TTClient, store: BlobStore, resource: str, entity: Dict, include: Optional[List[str]]):
    entity_id = entity.get("id")
    detailed = client.get_entity(resource, entity_id, include=include) if include else {"data": entity}
    store.upload_json(f"tt/enriched/{resource}/{entity_id}.json", detailed)

def fetch_and_store_raw(client: TTClient, store: BlobStore, resource: str, per_page: int, include: Optional[List[str]] = None, force: bool = False):
    for item in client.list_resource(resource, per_page=per_page):
        entity_id = item.get("id")
        raw_path = f"tt/raw/{resource}/{entity_id}.json"
        if (not force) and store.exists(raw_path):
            continue
        store.upload_json(raw_path, {"data": item})

def _download_stream(url: str):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "application/octet-stream")
        data = r.content
        return data, content_type

def refresh_candidate_files(client: TTClient, store: BlobStore, candidate: Dict, force: bool = False):
    cid = candidate.get("id")

    # 1) Essai via include=uploads (JSON:API included)
    uploads: List[Dict] = []
    try:
        doc = client.get_entity("candidates", cid, include=["uploads"])
        for inc in doc.get("included", []) or []:
            if inc.get("type") in ("uploads", "upload"):
                attrs = inc.get("attributes", {}) or {}
                url = attrs.get("url")
                if url:
                    uploads.append({"id": inc.get("id"), "url": url})
    except Exception as e:
        logger.debug(f"[{cid}] include=uploads not available: {e}")

    # 2) Fallback /uploads?filter[candidate]=<id>
    if not uploads:
        try:
            for up in client.list_uploads_by_candidate(cid, per_page=100):
                attrs = up.get("attributes", {}) or {}
                url = attrs.get("url")
                if url:
                    uploads.append({"id": up.get("id"), "url": url})
        except Exception as e:
            logger.debug(f"[{cid}] list_uploads_by_candidate failed: {e}")

    # 3) Dernier filet: anciennes clés (héritées de tes notebooks)
    attrs = candidate.get("attributes", {}) or {}
    for k in ("cv-url", "resume-url", "original-file-url"):
        if attrs.get(k):
            uploads.append({"id": k, "url": attrs[k]})

    # Téléchargement idempotent
    for up in uploads:
        url = up["url"]
        try:
            data, ctype = _download_stream(url)
        except Exception as e:
            logger.warning(f"[{cid}] Download failed {url}: {e}")
            continue
        ext = extension_from_mime(ctype, default=".bin")
        blob_path = f"tt/files/candidates/{cid}/uploads/upload_{up['id']}{ext}"
        if (not force) and store.exists(blob_path):
            continue
        store.upload_bytes(blob_path, data, ctype)

def enrich_resource(client: TTClient, store: BlobStore, state: StateStore, resource: str, per_page: int, max_workers: int, force: bool = False, include: Optional[List[str]] = None, with_files: bool = False):
    # 1) RAW (idempotent)
    fetch_and_store_raw(client, store, resource, per_page, include=None, force=force)

    # 2) ENRICH (concurrent)
    items = list(client.list_resource(resource, per_page=per_page))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(enrich_entity, client, store, resource, it, include) for it in items]
        for fut in as_completed(futures):
            _ = fut.result()

    # 3) FICHIERS candidats (uploads)
    if resource == "candidates" and with_files:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(refresh_candidate_files, client, store, it, force) for it in items]
            for fut in as_completed(futures):
                _ = fut.result()

    # 4) STATE
    st = state.load(resource)
    st["items"] = st.get("items", 0) + len(items)
    state.save(resource, st)
