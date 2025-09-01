
import os, sys, json, datetime, re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceNotFoundError

AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=absaifabrik;AccountKey=dHnt6TqjK6GYHvEjLTKenFdRHitSCByxcqenpCAvP/+GkY6XjHk7+BMfSpVuhbSpUi/5EfGq61CR+AStw0NiCA==;EndpointSuffix=core.windows.net"
AZURE_BLOB_CONTAINER="cvcompat"



# ========= Utils =========
def utc_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)

def safe_get(d: Dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
        if cur is None:
            return default
    return cur

def to_iso(ts: Optional[str]) -> Optional[str]:
    return ts if ts else None

def parse_dt(ts: Optional[str]) -> Optional[datetime.datetime]:
    if not ts: return None
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z","+00:00"))
    except Exception:
        return None

def days_between(a: Optional[str], b: Optional[str]) -> Optional[float]:
    da, db = parse_dt(a), parse_dt(b)
    if not (da and db): return None
    return round((db - da).total_seconds() / 86400.0, 3)

# ========= Azure =========
def get_container():
    container_name = AZURE_BLOB_CONTAINER
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

def download_json(container, path: str) -> Dict[str, Any]:
    return json.loads(container.get_blob_client(path).download_blob().readall())

def upload_text(container, path: str, text: str, overwrite: bool=False, content_type: str="application/json; charset=utf-8"):
    container.get_blob_client(path).upload_blob(
        text.encode("utf-8"), overwrite=overwrite,
        content_settings=ContentSettings(content_type=content_type)
    )

# ========= Normalisations simples =========
def norm_status(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.lower().strip()
    if s in {"active","draft","archived"}: return s
    return s

def norm_employment_type(s: Optional[str]) -> Optional[str]:
    if not s: return None
    return s.lower().strip()

def norm_employment_level(s: Optional[str]) -> Optional[str]:
    if not s: return None
    return s.lower().strip()

def norm_remote_status(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.lower().strip()
    if "remote" in s and "hybrid" not in s: return "remote"
    if "hybrid" in s: return "hybrid"
    if s in {"none","on-site","onsite"}: return "none"
    return s

# ========= Build job lookup from Bronze/jobs =========
def build_job_lookup(container, bronze_jobs_prefix: str) -> Dict[str, Dict[str, Any]]:
    paths = list_json_blobs(container, bronze_jobs_prefix)
    out: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        try:
            doc = download_json(container, p)
            data = doc.get("data") or {}
            if data.get("type") != "jobs":
                continue
            a = data.get("attributes") or {}
            included = doc.get("included") or []

            # locations (labels simples)
            locs: List[str] = []
            for inc in included:
                if inc.get("type") in {"locations","offices"}:
                    ia = inc.get("attributes") or {}
                    city = ia.get("city") or ia.get("name")
                    country = ia.get("country") or ""
                    label = city if city else None
                    if label and country and country.lower() not in label.lower():
                        label = f"{label}, {country}"
                    if label: locs.append(label)
            if locs:
                # dedup
                seen, uniq = set(), []
                for x in locs:
                    if x not in seen:
                        seen.add(x); uniq.append(x)
                locs = uniq

            # tags (si tableau)
            tags = a.get("tags") or []
            if isinstance(tags, list):
                tags = sorted({str(t).strip().lower() for t in tags if str(t).strip()})
            else:
                tags = []

            out[str(data.get("id"))] = {
                "job_title": a.get("title") or a.get("internal-name"),
                "job_status": norm_status(a.get("status") or a.get("human-status")),
                "job_employment_type": norm_employment_type(a.get("employment-type")),
                "job_employment_level": norm_employment_level(a.get("employment-level")),
                "job_language_code": a.get("language-code"),
                "job_remote_status": norm_remote_status(a.get("remote-status")),
                "job_created_at": to_iso(a.get("created-at")),
                "job_updated_at": to_iso(a.get("updated-at")),
                "job_start_date": a.get("start-date"),
                "job_end_date": a.get("end-date"),
                "job_locations": locs,
                "job_department": _extract_one_name(included, "departments") or _extract_one_name(included, "department"),
                "job_division": _extract_one_name(included, "divisions") or _extract_one_name(included, "division"),
                "job_recruiter_email": a.get("recruiter-email"),
                "job_tags": tags,
                # on n’inclut pas ici le body_html/texte pour garder ce fichier léger
            }
        except Exception as e:
            # on ignore les jobs illisibles
            continue
    return out

def _extract_one_name(included: List[Dict[str, Any]], type_name: str) -> Optional[str]:
    for inc in included or []:
        if inc.get("type") == type_name:
            ia = inc.get("attributes") or {}
            return ia.get("name") or ia.get("title")
    return None

# ========= Build application row =========
def build_application_row(doc: Dict[str, Any], job_lut: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    data = doc.get("data") or {}
    if data.get("type") != "job-applications":
        return None
    a = data.get("attributes") or {}
    rel = data.get("relationships") or {}
    inc = doc.get("included") or []

    candidate_id = safe_get(rel, ["candidate","data","id"])
    job_id = safe_get(rel, ["job","data","id"])
    stage_id = safe_get(rel, ["stage","data","id"])
    reject_reason_id = safe_get(rel, ["reject-reason","data","id"])

    # from included
    stage_name = None
    reject_reason_text = None
    for i in inc:
        if i.get("type") == "stages" and i.get("id") == stage_id:
            stage_name = safe_get(i, ["attributes", "name"])
        if i.get("type") == "reject-reasons" and i.get("id") == reject_reason_id:
            # ✅ Teamtailor met le libellé ici
            reject_reason_text = (
                safe_get(i, ["attributes", "reason"])  # <— la bonne clé
                or safe_get(i, ["attributes", "name"])
                or safe_get(i, ["attributes", "title"])
                or safe_get(i, ["attributes", "text"])
            )

    # Fallback si le texte est porté par l'application elle-même (certaines versions/exports)
    if not reject_reason_text:
        reject_reason_text = (
            safe_get(a, ["reject-reason-text"])
            or safe_get(a, ["reject_reason_text"])
            or safe_get(a, ["reject-reason"])  # par prudence
        )

    created_at = to_iso(a.get("created-at"))
    updated_at = to_iso(a.get("updated-at"))
    changed_stage_at = to_iso(a.get("changed-stage-at"))
    rejected_at = to_iso(a.get("rejected-at"))

    decision = "rejected" if rejected_at else "in_process"

    row: Dict[str, Any] = {
        # core app
        "application_id": data.get("id"),
        "candidate_id": candidate_id,
        "job_id": job_id,
        "status": a.get("status"),
        "stage_id": stage_id,
        "stage_name": stage_name,
        "created_at": created_at,
        "updated_at": updated_at,
        "changed_stage_at": changed_stage_at,
        "rejected_at": rejected_at,
        "reject_reason_id": reject_reason_id,
        "reject_reason_text": reject_reason_text,
        "source_site": a.get("referring-site"),
        "source_url": a.get("referring-url"),
        "sourced": a.get("sourced"),
        "cover_letter_present": bool(a.get("cover-letter")),
        "decision": decision,
        "timings": {
            "days_to_reject": days_between(created_at, rejected_at),
            "days_since_created": days_between(created_at, utc_iso()),
        },
    }

    # enrich job snapshot (aplati avec préfixe job_)
    if job_id and str(job_id) in job_lut:
        row.update(job_lut[str(job_id)])

    return row

# ========= Main =========
def run():
    load_dotenv()
    container = get_container()

    bronze_jobs_prefix = os.getenv("BRONZE_JOBS_PREFIX", "bronze/jobs/")
    bronze_apps_prefix = os.getenv("BRONZE_JOB_APPS_PREFIX", "bronze/job-applications/")
    silver_apps_prefix = os.getenv("SILVER_JOB_APPS_PREFIX", "silver/job-applications/")

    # 1) build job lookup once
    print("🧭 Construction du lookup jobs…")
    job_lut = build_job_lookup(container, bronze_jobs_prefix)
    print(f"   → {len(job_lut)} jobs indexés")

    apps_total = 0
    apps_with_job_id = 0
    apps_with_job_found = 0
    apps_with_job_missing = 0
    missing_job_ids = set()

    # 2) process all applications
    src_paths = list_json_blobs(container, bronze_apps_prefix)
    print(f"🔎 {len(src_paths)} job-applications Bronze trouvées sous '{bronze_apps_prefix}'")

    run_ts = utc_iso().replace(":", "-")
    out_path = f"{silver_apps_prefix}{run_ts}.jsonl"
    manifest_path = f"silver/_manifests/job-applications_{run_ts}.json"

    lines: List[str] = []
    ok = bad = 0

    for src in src_paths:
        try:
            doc = download_json(container, src)
            row = build_application_row(doc, job_lut)
            if not row:
                continue
            apps_total += 1
            job_id = row.get("job_id")
            if job_id:
                apps_with_job_id += 1
                if str(job_id) in job_lut:
                    apps_with_job_found += 1
                else:
                    apps_with_job_missing += 1
                    missing_job_ids.add(str(job_id))
            lines.append(jdump(row))
            ok += 1
        except Exception as e:
            bad += 1
            print(f"✗ Erreur application {src}: {e}", file=sys.stderr)

    if lines:
        upload_text(container, out_path, "\n".join(lines) + "\n", overwrite=False)

    manifest = {
        "run_ts": run_ts,
        "bronze_jobs_prefix": bronze_jobs_prefix,
        "bronze_job_applications_prefix": bronze_apps_prefix,
        "silver_job_applications_path": out_path,
        "counts": {
            "ok": ok,
            "failed": bad,
            "applications_total": apps_total,
            "applications_with_job_id": apps_with_job_id,
            "applications_with_job_found": apps_with_job_found,
            "applications_with_job_missing": apps_with_job_missing,
            "unique_missing_job_ids": len(missing_job_ids)
        },
        "missing_job_ids_sample": sorted(list(missing_job_ids))[:10]  # aperçu
    }
    upload_text(container, manifest_path, jdump(manifest), overwrite=False)

    print("\n=== FIN ===")
    print("\n=== COUVERTURE JOBS ===")
    print(f"Total applications         : {apps_total}")
    print(f"Applications avec job_id   : {apps_with_job_id}")
    print(f" → job trouvé dans lookup  : {apps_with_job_found}")
    print(f" → job manquant            : {apps_with_job_missing}")
    print(f"IDs de job manquants (apercu ≤50) : {manifest['missing_job_ids_sample']}")
    print(jdump(manifest))

if __name__ == "__main__":
    run()
