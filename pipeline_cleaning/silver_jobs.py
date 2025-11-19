#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, datetime, re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceNotFoundError
# --- NEW (si manquants) ---
import time, math
import requests

# =============== LLM Job Segmentation (Azure OpenAI) ===============
AZURE_OPENAI_ENDPOINT   = "https://saegus-openai-us.openai.azure.com"
AZURE_OPENAI_API_KEY    = "bd43498be60845caa72d6963645dd1e1"
AZURE_OPENAI_API_VER    = "2024-12-01-preview"
JOBS_LLM_DEPLOYMENT     = "gpt-4o"

# Toggle & throttle
JOBS_LLM_SEGMENT        = "1" # "1" pour activer
JOBS_LLM_RPM            = 30000  # appels/min max
_last_call_ts           = [0.0]


AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=absaifabrik;AccountKey=dHnt6TqjK6GYHvEjLTKenFdRHitSCByxcqenpCAvP/+GkY6XjHk7+BMfSpVuhbSpUi/5EfGq61CR+AStw0NiCA==;EndpointSuffix=core.windows.net"
AZURE_BLOB_CONTAINER="cvcompat"

JOB_JSON_SCHEMA_EXAMPLE = {
    "job": {
        "title": "string",
        "language": "fr|en|...",
        "sections": {
            "responsibilities": {"bullets": ["string"], "text": "string"},
            "requirements_must": {
                "skills": ["string"], "bullets": ["string"], "text": "string",
                "years_required": {"min": 0.0, "preferred": 0.0}
            },
            "requirements_nice": {"skills": ["string"], "bullets": ["string"], "text": "string"},
            "languages": [{"code": "fr", "level": "C1"}],
            "education": {"level": "bac+5/master|bac+3/licence|bac+2|phd|...", "fields": ["string"], "text": "string"},
            "contract": {"type": "permanent|contract|internship|apprenticeship|part-time|freelance", "text": "string"},
            "remote": {"type": "remote|hybrid|on-site", "text": "string"},
            "salary": {"min": 0.0, "max": 0.0, "currency": "EUR|USD|...", "period": "year|month", "raw": "string"},
            "benefits": {"bullets": ["string"], "text": "string"},
            "company": {"text": "string"}
        },
        "keywords": ["string"]
    }
}


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

def _throttle():
    if JOBS_LLM_RPM <= 0: return
    min_interval = 60.0 / JOBS_LLM_RPM
    since = time.time() - _last_call_ts[0]
    if since < min_interval:
        time.sleep(min_interval - since)
    _last_call_ts[0] = time.time()

def _truncate(txt: str, max_chars: int = 12000) -> str:
    if not isinstance(txt, str): return ""
    if len(txt) <= max_chars: return txt
    return txt[:max_chars]

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


#================LLM Segmenter (Azure OpenAI) ===============
def _llm_messages_for_job(title: str, body_text: str, language_code: Optional[str]) -> list:
    sys = (
        "You are an ATS Job Description segmenter. "
        "Extract a clean, structured JSON for the job below. "
        "Return STRICT JSON only. No commentary. "
        "Prefer the language of the job text (FR/EN). "
        "If a section is absent, return empty arrays/empty strings, not null."
    )
    user = {
        "task": "Segment this job into structured sections.",
        "json_contract": JOB_JSON_SCHEMA_EXAMPLE,
        "hints": [
            "Detect skills (hard/soft) in must vs nice.",
            "Infer languages (codes: fr,en,es,de,it,pt,ar,...) and CEFR levels if present (A1–C2).",
            "Extract years_required (min, preferred) if ranges (ex: 3–5 ans → min=3, preferred=5).",
            "Infer remote/contract if present in text.",
            "Detect benefits (tickets resto, mutuelle, etc.).",
            "Salary: extract min/max + currency + period; put original span in 'raw'.",
            "Education: map levels to categories (bac+2, bac+3/licence, bac+5/master, phd...).",
        ],
        "job_title": title or "",
        "job_language_hint": language_code or "",
        "job_body_text": _truncate(body_text, 12000),
    }
    return [
        {"role":"system","content":sys},
        {"role":"user","content":json.dumps(user, ensure_ascii=False)}
    ]

def _azure_chat_completions(endpoint: str, deployment: str, api_key: str, api_version: str, messages: list, max_tokens: int = 2000):
    """
    Appel direct REST Azure Chat Completions pour éviter les soucis de SDK.
    """
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {"Content-Type":"application/json", "api-key": api_key}
    payload = {
        "messages": messages,
        "temperature": 0,
        "n": 1,
        "max_tokens": max_tokens,
        # Guide JSON strict (Azure supporte response_format.type=json_object)
        "response_format": {"type": "json_object"}
    }
    _throttle()
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Azure Chat error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content

def _try_parse_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def segment_job_with_llm(title: Optional[str], body_text: Optional[str], language_code: Optional[str]) -> dict:
    """
    Retourne un dict (toujours) avec:
      - "ok": bool
      - "data": dict JSON (ou {})
      - "error": str (si échec)
    """
    if JOBS_LLM_SEGMENT not in {"1","true","TRUE","yes","YES"}:
        return {"ok": False, "data": {}, "error": "LLM disabled via JOBS_LLM_SEGMENT"}
    if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and JOBS_LLM_DEPLOYMENT):
        return {"ok": False, "data": {}, "error": "Azure OpenAI env vars missing"}

    if not body_text:
        return {"ok": False, "data": {}, "error": "empty body_text"}

    msgs = _llm_messages_for_job(title or "", body_text or "", language_code)
    try:
        raw = _azure_chat_completions(
            endpoint=AZURE_OPENAI_ENDPOINT.rstrip("/"),
            deployment=JOBS_LLM_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VER,
            messages=msgs,
            max_tokens=2200,
        )
        data = _try_parse_json(raw)
        if data is None:
            # tentative de réparation simple
            fix_msgs = msgs + [{"role":"user","content":"Return the SAME content as STRICT valid JSON only. No prose."}]
            raw2 = _azure_chat_completions(
                AZURE_OPENAI_ENDPOINT.rstrip("/"),
                JOBS_LLM_DEPLOYMENT,
                AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_API_VER,
                fix_msgs, max_tokens=2200
            )
            data = _try_parse_json(raw2)
            if data is None:
                return {"ok": False, "data": {}, "error": "invalid JSON from LLM"}
        return {"ok": True, "data": data, "error": ""}
    except Exception as e:
        return {"ok": False, "data": {}, "error": str(e)[:500]}

def _listify(x):
    if x is None: return []
    if isinstance(x, list): return [t for t in x if isinstance(t, str) and t.strip()]
    if isinstance(x, str): return [x] if x.strip() else []
    return []

def _get(d, path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict): return default
        cur = cur.get(p)
        if cur is None: return default
    return cur

def _float_or_none(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except: return None

def flatten_job_llm_segments(seg_json: dict) -> dict:
    """
    Aplati quelques champs utiles depuis le JSON LLM.
    """
    out = {}
    sec = _get(seg_json, ["job","sections"], {}) or {}

    # Lists
    out["job_requirements_list_llm"]   = _listify(_get(sec, ["requirements_must","bullets"], []))
    out["job_responsibilities_list_llm"]= _listify(_get(sec, ["responsibilities","bullets"], []))
    out["job_nice_list_llm"]           = _listify(_get(sec, ["requirements_nice","bullets"], []))
    out["job_benefits_list_llm"]       = _listify(_get(sec, ["benefits","bullets"], []))

    # Text blocks
    out["job_requirements_text_llm"]   = _get(sec, ["requirements_must","text"])
    out["job_responsibilities_text_llm"]= _get(sec, ["responsibilities","text"])
    out["job_nice_text_llm"]           = _get(sec, ["requirements_nice","text"])
    out["job_benefits_text_llm"]       = _get(sec, ["benefits","text"])
    out["job_languages_text_llm"]      = ", ".join([json.dumps(x, ensure_ascii=False) for x in (_get(sec,["languages"],[]) or [])]) or None
    out["job_salary_text_llm"]         = _get(sec, ["salary","raw"])
    out["job_contract_text_llm"]       = _get(sec, ["contract","text"])
    out["job_remote_text_llm"]         = _get(sec, ["remote","text"])
    out["job_education_text_llm"]      = _get(sec, ["education","text"])

    # Structured: skills, languages, years, salary, contract/remote/education
    out["required_skills_must_llm"] = _listify(_get(sec, ["requirements_must","skills"], []))
    out["required_skills_plus_llm"] = _listify(_get(sec, ["requirements_nice","skills"], []))

    langs = _get(sec, ["languages"], []) or []
    # store list of codes; keep full JSON separately
    codes = []
    for x in langs:
        if isinstance(x, dict):
            code = x.get("code")
            if isinstance(code, str) and code.strip():
                codes.append(code.lower().strip())
    out["required_languages_llm"] = sorted(list(set(codes))) if codes else []

    yrs = _get(sec, ["requirements_must","years_required"], {}) or {}
    out["job_years_required_min_llm"]       = _float_or_none(yrs.get("min"))
    out["job_years_required_preferred_llm"] = _float_or_none(yrs.get("preferred"))

    sal = _get(sec, ["salary"], {}) or {}
    out["job_salary_min_llm"]      = _float_or_none(sal.get("min"))
    out["job_salary_max_llm"]      = _float_or_none(sal.get("max"))
    out["job_salary_currency_llm"] = sal.get("currency")
    out["job_salary_period_llm"]   = sal.get("period")

    out["job_contract_type_llm"]   = _get(sec, ["contract","type"])
    out["job_remote_type_llm"]     = _get(sec, ["remote","type"])

    edu = _get(sec, ["education","level"])
    out["education_level_required_llm"] = edu

    # Keywords
    out["keywords_job_llm"] = _listify(_get(seg_json, ["job","keywords"], []))

    return out

def llm_env_ok() -> tuple[bool, dict]:
    info = {
        "endpoint": bool(AZURE_OPENAI_ENDPOINT),
        "api_key": bool(AZURE_OPENAI_API_KEY),
        "api_version": bool(AZURE_OPENAI_API_VER),
        "deployment": bool(JOBS_LLM_DEPLOYMENT),
        "segment_toggle": JOBS_LLM_SEGMENT in {"1","true","TRUE","yes","YES"},
    }
    ok = all([info["endpoint"], info["api_key"], info["deployment"], info["segment_toggle"]])
    return ok, info

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
            # --- NEW: LLM segmentation (optionnelle) ---
    llm_res = segment_job_with_llm(title, body_text, language_code)
    if llm_res.get("ok"):
        llm_data = llm_res.get("data") or {}
        row["job_llm_segments"] = llm_data  # JSON brut complet
        # aplatissement utile
        flat = flatten_job_llm_segments(llm_data)
        row.update(flat)
        row["job_llm_error"] = None
    else:
        row["job_llm_segments"] = None
        row["job_llm_error"] = llm_res.get("error")



    return row

# =============== Main ===============
def run():
    load_dotenv()
    container = get_container()

    bronze_jobs_prefix = os.getenv("BRONZE_JOBS_PREFIX", "bronze/jobs/")
    silver_jobs_prefix = os.getenv("SILVER_JOBS_PREFIX", "silver/jobs/")

    src_paths = list_json_blobs(container, bronze_jobs_prefix)
    print(f"🔎 {len(src_paths)} jobs Bronze trouvés sous '{bronze_jobs_prefix}'")
    ok_env, info = llm_env_ok()
    print("LLM env:", {k: ("OK" if v else "MISSING") for k,v in info.items()})


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
    llm_ok = sum(1 for line in lines if '"job_llm_segments":' in line and '"job_llm_segments": null' not in line)
    llm_err = sum(1 for line in lines if '"job_llm_error":' in line and '"job_llm_error": null' not in line)

    manifest["llm_stats"] = {"segments_ok": llm_ok, "segments_error": llm_err}
    manifest["llm_env"] = info  # utile tant que tu débogues

    print("\n=== FIN ===")
    print(jdump(manifest))

if __name__ == "__main__":
    run()
