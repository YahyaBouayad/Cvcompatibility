import os, sys, json, datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient,ContentSettings
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError, HttpResponseError
import re

AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=absaifabrik;AccountKey=dHnt6TqjK6GYHvEjLTKenFdRHitSCByxcqenpCAvP/+GkY6XjHk7+BMfSpVuhbSpUi/5EfGq61CR+AStw0NiCA==;EndpointSuffix=core.windows.net"
AZURE_BLOB_CONTAINER="cvcompat"
# lecture depuis blob
SOURCE_CANDIDATES_PREFIX="tt/enriched/candidates/"
SOURCE_JOBS_PREFIX="tt/enriched/jobs/"
SOURCE_APPLICATIONS_PREFIX="tt/enriched/job-applications/"
# Destinations Bronze
BRONZE_CANDIDATES_PREFIX="bronze/candidates/"
BRONZE_JOBS_PREFIX="bronze/jobs/"
BRONZE_APPLICATIONS_PREFIX="bronze/job-applications/"

# Utils
# ==========================
def utc_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)
# --- helpers metadata ---


_ALLOWED_KEY = re.compile(r'[^a-z0-9-]')  # lettres/chiffres/tiret

def sanitize_metadata(meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not meta:
        return None
    clean: Dict[str, str] = {}
    for k, v in meta.items():
        if k is None:
            continue
        kk = _ALLOWED_KEY.sub('-', str(k).strip().lower())
        if not kk:
            continue
        vv = '' if v is None else str(v)
        # Values must be ASCII and single-line
        vv = vv.encode('ascii', 'ignore').decode('ascii')
        vv = vv.replace('\r', ' ').replace('\n', ' ')
        clean[kk] = vv
    return clean or None


# ==========================
# Azure helpers
# ==========================
def get_container():
    container_name = os.getenv("AZURE_BLOB_CONTAINER", "cvcompat")
    conn_str = AZURE_STORAGE_CONNECTION_STRING
    if not conn_str:
        print("❌ AZURE_STORAGE_CONNECTION_STRING manquante.", file=sys.stderr); sys.exit(1)
    bsc = BlobServiceClient.from_connection_string(conn_str)
    cc = bsc.get_container_client(container_name)
    try:
        cc.get_container_properties()
    except ResourceNotFoundError:
        cc.create_container()
    return cc

def list_json_blobs(container, prefix: str) -> List[str]:
    return [b.name for b in container.list_blobs(name_starts_with=prefix) if b.name.lower().endswith(".json")]

def download_bytes(container, path: str) -> bytes:
    return container.get_blob_client(path).download_blob().readall()

def blob_exists(container, path: str) -> bool:
    try:
        container.get_blob_client(path).get_blob_properties()
        return True
    except ResourceNotFoundError:
        return False

def upload_bytes(container, path: str, content: bytes,
                 metadata: Optional[Dict[str, Any]] = None,
                 overwrite: bool = False) -> bool:
    from azure.storage.blob import ContentSettings
    bc = container.get_blob_client(path)
    safe_meta = sanitize_metadata(metadata)
    try:
        bc.upload_blob(
            content,
            overwrite=overwrite,
            metadata=safe_meta,
            content_settings=ContentSettings(content_type="application/json; charset=utf-8"),
        )
        return True
    except HttpResponseError as e:
        # Si Azure refuse les metadata (InvalidMetadata), on réessaie sans metadata
        if "InvalidMetadata" in str(e):
            bc.upload_blob(
                content,
                overwrite=overwrite,
                content_settings=ContentSettings(content_type="application/json; charset=utf-8"),
            )
            return True
        # autre erreur -> échec réel
        return False


# ==========================
# Validators & id extractors
# ==========================
def _is_type(doc: Dict[str, Any], tt_type: str) -> bool:
    try:
        return doc.get("data", {}).get("type") == tt_type and "id" in doc.get("data", {})
    except Exception:
        return False

def _id(doc: Dict[str, Any]) -> str:
    return str(doc["data"]["id"])

# ==========================
# Generic ingestion
# ==========================
def ingest_entity(container, source_prefix: str, bronze_prefix: str, expected_type: str) -> Dict[str,int]:
    run_ts = utc_iso().replace(":", "-")
    src_paths = list_json_blobs(container, source_prefix)
    print(f"\n=== {expected_type.upper()} ===")
    print(f"🔎 {len(src_paths)} fichier(s) trouvés sous '{source_prefix}'")

    ingested, skipped, bad = 0, 0, 0
    manifest_lines: List[str] = []

    for src in src_paths:
        try:
            raw = download_bytes(container, src)
            doc = json.loads(raw)

            if not _is_type(doc, expected_type):
                print(f"  • skip (type != '{expected_type}') : {src}")
                skipped += 1
                continue

            eid = _id(doc)
            dest = f"{bronze_prefix}{eid}.json"

            if blob_exists(container, dest):
                print(f"  • existe déjà → skip : {dest}")
                skipped += 1
                continue

            # on fige le timestamp avant upload (sans dépendre des metadata)
            ingested_ts = utc_iso()

            meta = {
                "entity": expected_type,  # ex: "job-applications"
                "id": eid,
                "ingested": ingested_ts,  # clé safe (et on ne la relira pas)
                "src": src,
            }

            ok = upload_bytes(container, dest, raw, metadata=meta, overwrite=False)
            if not ok:
                bad += 1
                print(f"  ✗ échec upload (réel) : {src}")
                continue

            ingested += 1
            # ⚠️ manifest utilise la variable ingested_ts, PAS meta["..."]
            manifest_lines.append(jdump({
                "id": eid,
                "source_blob": src,
                "bronze_blob": dest,
                "bronze_ingested_at": ingested_ts
            }))
            print(f"  ✓ ingéré → {dest}")

        except Exception as e:
            bad += 1
            print(f"  ✗ échec {src} : {e}", file=sys.stderr)

    # manifest
    if manifest_lines:
        manifest_path = f"{bronze_prefix}_manifest/{run_ts}.jsonl"
        upload_bytes(container, manifest_path, ("\n".join(manifest_lines) + "\n").encode("utf-8"))
        print(f"🗒️  manifest écrit → {manifest_path}")

    print(f"Résumé {expected_type}: ingérés={ingested} | skippés={skipped} | échecs={bad}")
    return {"ingested": ingested, "skipped": skipped, "failed": bad}

# ==========================
# Main
# ==========================
def main():
    load_dotenv()
    c = get_container()

    # prefixes
    src_cand = os.getenv("SOURCE_CANDIDATES_PREFIX", "tt/enriched/candidates/")
    src_jobs = os.getenv("SOURCE_JOBS_PREFIX", "tt/enriched/jobs/")
    src_apps = os.getenv("SOURCE_APPLICATIONS_PREFIX", "tt/enriched/job-applications/")

    br_cand = os.getenv("BRONZE_CANDIDATES_PREFIX", "bronze/candidates/")
    br_jobs = os.getenv("BRONZE_JOBS_PREFIX", "bronze/jobs/")
    br_apps = os.getenv("BRONZE_APPLICATIONS_PREFIX", "bronze/job-applications/")

    print("🚀 Démarrage ingestion Bronze (candidates, jobs, job-applications)")

    s1 = ingest_entity(c, src_cand, br_cand, "candidates")
    s2 = ingest_entity(c, src_jobs, br_jobs, "jobs")
    s3 = ingest_entity(c, src_apps, br_apps, "job-applications")

    total = {k: s1.get(k,0)+s2.get(k,0)+s3.get(k,0) for k in ["ingested","skipped","failed"]}
   
    print("\n=== RÉCAP GLOBAL ===")
    print(jdump(total))

if __name__ == "__main__":
    main()