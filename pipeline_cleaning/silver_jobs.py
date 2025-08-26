#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, datetime, re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceNotFoundError

AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=absaifabrik;AccountKey=dHnt6TqjK6GYHvEjLTKenFdRHitSCByxcqenpCAvP/+GkY6XjHk7+BMfSpVuhbSpUi/5EfGq61CR+AStw0NiCA==;EndpointSuffix=core.windows.net"
AZURE_BLOB_CONTAINER="cvcompat"




# =============== Utils ===============
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

def word_count(txt: Optional[str]) -> Optional[int]:
    if not txt: return None
    return len(re.findall(r"\w+", txt, flags=re.UNICODE))

# =============== HTML → text ===============
def html_to_text(html: Optional[str]) -> Optional[str]:
    if not html: return None
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        # normaliser les espaces/sauts
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n\s*", "\n\n", text).strip()
        return text or None
    except Exception:
        # Fallback minimal si bs4 indisponible
        txt = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n\s*\n\s*", "\n\n", txt).strip()
        return txt or None

# =============== Azure helpers ===============
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

def upload_text(container, path: str, text: str, overwrite: bool = False, content_type: str = "application/json; charset=utf-8"):
    container.get_blob_client(path).upload_blob(
        text.encode("utf-8"), overwrite=overwrite,
        content_settings=ContentSettings(content_type=content_type)
    )

# =============== Normalisation valeurs ===============
def norm_status(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.lower().strip()
    # Teamtailor: active, draft, archived (ou human-status séparé)
    if s in {"active","draft","archived"}: return s
    return s

def norm_employment_type(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.lower().strip()
    mapping = {
        "full_time": "permanent",
        "permanent": "permanent",
        "contract": "contract",
        "internship": "internship",
        "apprenticeship": "internship",
        "freelance": "freelance",
        "part_time": "contract"
    }
    return mapping.get(s, s)

def norm_employment_level(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.lower().strip()
    mapping = {
        "junior": "junior",
        "mid": "mid",
        "senior": "senior",
        "lead": "lead",
        "manager": "manager",
        "none": "none"
    }
    return mapping.get(s, s)

def norm_remote_status(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.lower().strip()
    if "remote" in s and "hybrid" not in s: return "remote"
    if "hybrid" in s: return "hybrid"
    if s in {"none","on-site","onsite"}: return "none"
    return s

def to_bool_req(x: Optional[str]) -> Optional[bool]:
    if x is None: return None
    s = str(x).lower().strip()
    # Teamtailor: "required" | "optional" | "not-allowed"...
    if s in {"required","obligatoire"}: return True
    if s in {"optional","facultatif"}: return False
    return None

# =============== Extractors ===============
def extract_locations(included: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for inc in included or []:
        if inc.get("type") in {"locations","offices"}:
            a = inc.get("attributes") or {}
            # city/name/country
            city = a.get("city") or a.get("name")
            country = a.get("country") or ""
            label = city if city else None
            if label and country and country.lower() not in label.lower():
                label = f"{label}, {country}"
            if label:
                out.append(label)
    # dédoublonner
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

def extract_one_name(included: List[Dict[str, Any]], type_name: str) -> Optional[str]:
    for inc in included or []:
        if inc.get("type") == type_name:
            a = inc.get("attributes") or {}
            return a.get("name") or a.get("title")
    return None

def extract_tags(attributes: Dict[str, Any]) -> List[str]:
    tags = attributes.get("tags") or []
    if isinstance(tags, list):
        return sorted({str(t).strip().lower() for t in tags if str(t).strip()})
    return []

def build_job_row(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = doc.get("data") or {}
    if data.get("type") != "jobs":
        return None
    a = data.get("attributes") or {}
    included = doc.get("included") or []

    title = a.get("title") or a.get("internal-name")
    status = norm_status(a.get("status") or a.get("human-status"))
    employment_type = norm_employment_type(a.get("employment-type"))
    employment_level = norm_employment_level(a.get("employment-level"))
    language_code = a.get("language-code")
    remote_status = norm_remote_status(a.get("remote-status"))

    body_html = a.get("body")
    body_text = html_to_text(body_html)
    wc = word_count(body_text)

    # requirements
    req = {
        "resume_required": to_bool_req(a.get("resume-requirement")),
        "cover_letter_required": to_bool_req(a.get("cover-letter-requirement")),
        "phone_required": to_bool_req(a.get("phone-requirement")),
    }

    # locations / department / division depuis included
    locations = extract_locations(included)
    department = extract_one_name(included, "departments") or extract_one_name(included, "department")
    division = extract_one_name(included, "divisions") or extract_one_name(included, "division")

    # tags
    tags = extract_tags(a)

    created_at = to_iso(a.get("created-at"))
    updated_at = to_iso(a.get("updated-at"))
    start_date = a.get("start-date")
    end_date = a.get("end-date")

    # indicateurs simples
    is_active = (status == "active")
    age_days = None
    try:
        if created_at:
            dt = datetime.datetime.fromisoformat(created_at.replace("Z","+00:00"))
            age_days = (datetime.datetime.now(datetime.timezone.utc) - dt).days
    except Exception:
        age_days = None

    row = {
        "job_id": data.get("id"),
        "title": title,
        "status": status,
        "employment_type": employment_type,
        "employment_level": employment_level,
        "language_code": language_code,
        "remote_status": remote_status,
        "created_at": created_at,
        "updated_at": updated_at,
        "start_date": start_date,
        "end_date": end_date,
        "locations": locations,
        "department": department,
        "division": division,
        "recruiter_email": a.get("recruiter-email"),
        "tags": tags,
        "requirements": req,
        "body_html": body_html,         # utile debug (optionnel)
        "body_text": body_text,         # pour embeddings / recherche
        "body_word_count": wc,
        "has_description": bool(body_text),
        "manifold": {
            "is_active": is_active,
            "age_days": age_days
        }
    }
    return row

# =============== Main ===============
def run():
    load_dotenv()
    container = get_container()

    bronze_jobs_prefix = os.getenv("BRONZE_JOBS_PREFIX", "bronze/jobs/")
    silver_jobs_prefix = os.getenv("SILVER_JOBS_PREFIX", "silver/jobs/")

    src_paths = list_json_blobs(container, bronze_jobs_prefix)
    print(f"🔎 {len(src_paths)} jobs Bronze trouvés sous '{bronze_jobs_prefix}'")

    run_ts = utc_iso().replace(":", "-")
    out_path = f"{silver_jobs_prefix}{run_ts}.jsonl"
    manifest_path = f"silver/_manifests/jobs_{run_ts}.json"

    lines: List[str] = []
    ok = bad = 0

    for src in src_paths:
        try:
            doc = download_json(container, src)
            row = build_job_row(doc)
            if not row:
                continue
            lines.append(jdump(row))
            ok += 1
        except Exception as e:
            bad += 1
            print(f"✗ Erreur job {src}: {e}", file=sys.stderr)

    # Écrit le JSONL unique (tous les jobs)
    if lines:
        upload_text(container, out_path, "\n".join(lines) + "\n", overwrite=False)

    # Manifest
    manifest = {
        "run_ts": run_ts,
        "bronze_jobs_prefix": bronze_jobs_prefix,
        "silver_jobs_path": out_path,
        "counts": {"ok": ok, "failed": bad}
    }
    upload_text(container, manifest_path, jdump(manifest), overwrite=False)

    print("\n=== FIN ===")
    print(jdump(manifest))

if __name__ == "__main__":
    run()
