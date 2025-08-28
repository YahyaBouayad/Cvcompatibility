from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set
from pathlib import Path
import logging

from teamtailor import TTClient, TTError
from storage import BlobStore
from state import StateStore
from mime_utils import extension_from_mime
import config as cfg

logger = logging.getLogger("cvcompat.enrich")

# Progress UI (optionnel)
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False

# ===== Helpers =====
def _safe_slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-","_",".") else "-" for ch in name)[:80]

def _ext_from_filename_or_mime(filename: str, content_type: str) -> str:
    ext = Path(filename).suffix
    if not ext:
        ext = extension_from_mime(content_type, default=".bin")
    return ext

def _should_skip_file(store: BlobStore, blob_path: str, remote_updated_at: str, force_files: bool) -> bool:
    if force_files:
        return False
    if not store.exists(blob_path):
        return False
    meta = store.get_blob_metadata(blob_path) or {}
    return meta.get("tt_updated_at") == (remote_updated_at or "")

# ===== Core =====
def enrich_entity(client: TTClient, store: BlobStore, resource: str, entity: Dict, include: Optional[List[str]]):
    entity_id = entity.get("id")
    detailed = client.get_entity(resource, entity_id, include=include) if include else {"data": entity}
    store.upload_json(f"tt/enriched/{resource}/{entity_id}.json", detailed)

def fetch_and_store_raw(client: TTClient, store: BlobStore, resource: str, per_page: int,
                        include: Optional[List[str]] = None, force: bool = False) -> List[Dict]:
    out: List[Dict] = []
    for item in client.list_resource(resource, per_page=per_page):
        entity_id = item.get("id")
        raw_path = f"tt/raw/{resource}/{entity_id}.json"
        if (not force) and store.exists(raw_path):
            out.append(item)
            continue
        store.upload_json(raw_path, {"data": item})
        out.append(item)
    return out

def refresh_candidate_files(client: TTClient, store: BlobStore, candidate_entity: Dict, force_files: bool = False):
    """
    Nouvelle logique "attributs du candidat" (rapide) :
      1) Vérifie d'abord dans le Blob si 'resume' et 'original' existent déjà.
         - Si les deux existent et force_files=False -> SKIP tout appel API.
      2) Sinon, un seul GET /candidates/{id} pour récupérer les URLs.
      3) Télécharge uniquement ce qui manque et uploade :
         tt/files/candidates/{candidate_id}/resume.<ext>
         tt/files/candidates/{candidate_id}/original.<ext>
    """
    candidate_id = str(candidate_entity.get("id"))
    base_prefix = f"tt/files/candidates/{candidate_id}/"

    def _suffix_exists(suffix: str) -> bool:
        # On teste l'existence par préfixe: resume* ou original* (ext variable)
        for _ in store.list_prefix(base_prefix + suffix):
            return True
        return False

    # --- 1) Pré-check Blob ---
    resume_exists = _suffix_exists("resume")
    original_exists = _suffix_exists("original")

    if not force_files and resume_exists and original_exists:
        return  # tout est déjà présent, on évite tout appel API

    # --- 2) Récupérer les URLs (1 seul appel API) ---
    links = client.get_candidate_resume_links(candidate_id)
    resume_url = links.get("resume")
    original_url = links.get("original")

    # --- 3) Télécharger/Uploader uniquement ce qui manque ---
    def _download_and_store(url: str, suffix: str):
        if not url:
            return
        if not force_files and _suffix_exists(suffix):
            return  # recheck protect (cas course vers Blob)

        try:
            content, content_type = client.download_content(url)
        except TTError:
            # on tente pas d'autre endpoint ici, stratégie "attributs only"
            return

        # extension depuis le content-type
        ext = extension_from_mime(content_type, default=".bin")
        blob_path = f"{base_prefix}{suffix}{ext}"
        store.upload_bytes(
            path=blob_path,
            data=content,
            content_type=content_type,
            metadata={
                "tt_source": "candidate.attributes",
                "tt_candidate_id": candidate_id,
                "tt_suffix": suffix
            },
            overwrite=True
        )

    if not resume_exists or force_files:
        _download_and_store(resume_url, "resume")
    if not original_exists or force_files:
        _download_and_store(original_url, "original")


def enrich_resource(
    client: TTClient,
    store: BlobStore,
    state: StateStore,
    resource: str,
    per_page: int,
    max_workers: int,
    include: Optional[List[str]] = None,
    force: bool = False,
    with_files: bool = False,
    force_files: bool = False,
):
    # 1) RAW -> items (un seul listing)
    items: List[Dict] = fetch_and_store_raw(client, store, resource, per_page=per_page, include=include, force=force)

    # 2) ENRICHED en parallèle (barre si rich)
    def _run_enrich(it):
        try:
            enrich_entity(client, store, resource, it, include)
        except Exception as e:
            logger.exception(f"[{resource}] enrich worker failed: {e}")

    if _HAS_RICH:
        cols = [SpinnerColumn(), TextColumn(f"[bold]{resource}: enrich"), BarColumn(),
                TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                TextColumn("ETA"), TimeRemainingColumn()]
        with Progress(*cols) as prog:
            t = prog.add_task(f"{resource}: enrich", total=len(items))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for _ in as_completed([ex.submit(_run_enrich, it) for it in items]):
                    prog.advance(t)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for fut in as_completed([ex.submit(_run_enrich, it) for it in items]):
                try:
                    fut.result()
                except Exception as e:
                    logger.exception(f"[{resource}] enrich worker failed: {e}")

    # 3) Files candidats (pool dédié) — barre si rich
    if resource == "candidates" and with_files:
        def _run_files(it):
            try:
                refresh_candidate_files(client, store, it, force_files)
            except Exception as e:
                logger.exception(f"[candidates/files] worker failed: {e}")

        pool_size = max(cfg.FILES_MAX_WORKERS, max_workers)

        if _HAS_RICH:
            cols = [SpinnerColumn(), TextColumn("[bold]candidates: files"), BarColumn(),
                    TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                    TextColumn("ETA"), TimeRemainingColumn()]
            with Progress(*cols) as prog:
                t = prog.add_task("candidates: files", total=len(items))
                with ThreadPoolExecutor(max_workers=pool_size) as ex:
                    for _ in as_completed([ex.submit(_run_files, it) for it in items]):
                        prog.advance(t)
        else:
            with ThreadPoolExecutor(max_workers=pool_size) as ex:
                for fut in as_completed([ex.submit(_run_files, it) for it in items]):
                    try:
                        fut.result()
                    except Exception as e:
                        logger.exception(f"[candidates/files] worker failed: {e}")

    # 4) STATE
    st = state.load(resource)
    st["items"] = st.get("items", 0) + len(items)
    state.save(resource, st)

# ===== Backfill jobs depuis job-applications =====
def _extract_job_id_from_application(app: Dict) -> Optional[str]:
    rel = (app.get("relationships") or {}).get("job") or {}
    data = rel.get("data") or {}
    jid = data.get("id")
    if jid:
        return str(jid)
    for key in ("job_id","jobId","job-id","attributes.job-id","attributes.job_id"):
        cur = app
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        if cur:
            return str(cur)
    return None

def collect_job_ids_from_job_applications(client: TTClient, per_page: int) -> Set[str]:
    job_ids: Set[str] = set()
    for app in client.list_resource("job-applications", per_page=per_page):
        jid = _extract_job_id_from_application(app)
        if not jid:
            try:
                rel = client.get_related("job-applications", app["id"], "job")
                data = rel.get("data") or {}
                jid = data.get("id")
            except Exception:
                jid = None
        if jid:
            job_ids.add(str(jid))
    return job_ids

def backfill_jobs_from_jobapps(
    client: TTClient,
    store: BlobStore,
    state: StateStore,
    per_page: int,
    max_workers: int,
    include: Optional[List[str]] = None,
    force: bool = False,
):
    job_ids = collect_job_ids_from_job_applications(client, per_page=per_page)
    logger.info(f"[jobs/backfill] {len(job_ids)} job_ids collectés depuis job-applications")

    def _process_job(jid: str):
        doc = client.get_entity("jobs", jid, include=include) if include else client.get_entity("jobs", jid)
        raw_path = f"tt/raw/jobs/{jid}.json"
        if force or not store.exists(raw_path):
            store.upload_json(raw_path, {"data": (doc.get("data") or {})})
        store.upload_json(f"tt/enriched/jobs/{jid}.json", doc)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed([ex.submit(_process_job, jid) for jid in sorted(job_ids)]):
            try:
                fut.result()
            except Exception as e:
                logger.exception(f"[jobs/backfill] worker failed: {e}")

    st = state.load("jobs")
    st["items"] = st.get("items", 0) + len(job_ids)
    state.save("jobs", st)
    logger.info(f"[jobs/backfill] terminé — {len(job_ids)} jobs enrichis")
