#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GOLD builder (A..G sans ML) — 1 ligne = 1 application.

- Lit les silvers (applications, candidates, jobs) depuis Azure Blob
- Exploite la segmentation du silver candidates
- Features d'activités depuis 'activities_summary' (ou 'activities'), + texte notes RH
- Concatène des vues texte CV + fallbacks (role/description/profil)
- Mappage Job élargi + nettoyage (locations/tags/department)
- Fallbacks Candidat (titre & localisation depuis CV)
- Dates normalisées en UTC 'Z', stage normalisé
- Filtre: garde uniquement les candidatures avec has_segments=True (défaut)
- Ajouts: rejection_reason (incl. reject_reason_text), rejected_..._at, is_hired, is_rejected, application_outcome (incl. decision)
- Écrit: gold/applications_gold_{ts}.parquet + .jsonl + latest.*
"""

import os, io, re, json, argparse, sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient


# ============== Azure Blob helpers ==============

def make_blob_service_client() -> BlobServiceClient:
    load_dotenv()
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    account_url = os.getenv("BLOB_ACCOUNT_URL")
    sas = os.getenv("BLOB_SAS_TOKEN")
    if conn_str:
        return BlobServiceClient.from_connection_string(conn_str)
    if account_url and sas:
        return BlobServiceClient(account_url=account_url, credential=sas)
    raise RuntimeError("Provide AZURE_STORAGE_CONNECTION_STRING or (BLOB_ACCOUNT_URL + BLOB_SAS_TOKEN).")

def list_blobs(container_client, prefix: str):
    return [b for b in container_client.list_blobs(name_starts_with=prefix)]

def _blob_name(b) -> str:
    return b["name"] if isinstance(b, dict) else b.name

def _blob_last_modified(b):
    return b["last_modified"] if isinstance(b, dict) else b.last_modified

def download_blob_text(container_client, blob_name: str) -> str:
    return container_client.download_blob(blob_name).readall().decode("utf-8", errors="ignore")

def upload_bytes(container_client, blob_name: str, data: bytes, overwrite: bool = True) -> None:
    container_client.upload_blob(name=blob_name, data=data, overwrite=overwrite)


# ============== JSONL / Parquet helpers ==============

def jsonl_to_df(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s: 
            continue
        try:
            rows.append(json.loads(s))
        except Exception:
            # petites réparations usuelles (virgules traînantes)
            try:
                s2 = s.replace(",}", "}").replace(",]", "]")
                rows.append(json.loads(s2))
            except Exception:
                continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def read_latest_jsonl_df(container_client, prefix: str) -> pd.DataFrame:
    blobs = list_blobs(container_client, prefix)
    if not blobs:
        return pd.DataFrame()
    blobs_sorted = sorted(blobs, key=lambda b: _blob_last_modified(b), reverse=True)
    chosen = _blob_name(blobs_sorted[0])
    text = download_blob_text(container_client, chosen)
    return jsonl_to_df(text)

def to_jsonable(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, pd.Timestamp):
        return v.isoformat().replace("+00:00", "Z")
    if isinstance(v, datetime):
        s = v.astimezone(timezone.utc).isoformat()
        return s.replace("+00:00", "Z")
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v

def df_to_jsonl_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    for _, row in df.iterrows():
        obj = {k: to_jsonable(v) for k, v in row.to_dict().items()}
        buf.write(json.dumps(obj, ensure_ascii=False))
        buf.write("\n")
    return buf.getvalue().encode("utf-8")

def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, preserve_index=False)
    sink = io.BytesIO()
    pq.write_table(table, sink, compression="snappy")
    return sink.getvalue()


# ============== Normalisation utils ==============

def normalize_str(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).replace("\xa0", " ").strip()
    return s if s != "" else None

def to_iso(dt: Optional[Any]) -> Optional[str]:
    if dt is None or dt == "":
        return None
    try:
        ts = pd.to_datetime(dt, utc=True)
        return ts.isoformat().replace("+00:00", "Z")
    except Exception:
        return None

def pick(df: pd.DataFrame, cols: List[str], default=None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], dtype="object")
    out = None
    for c in cols:
        if c in df.columns:
            s = df[c]
            out = s if out is None else out.combine_first(s)
    return out if out is not None else pd.Series([default]*len(df), index=df.index)

def html_to_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        txt = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
        txt = re.sub(r"</p\s*>", "\n\n", txt, flags=re.I)
        txt = re.sub(r"<[^>]+>", "", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return normalize_str(txt)
    except Exception:
        return normalize_str(s)

def location_to_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, dict):
        cities = [v.get(k) for k in ["city","name","label","town","locality"] if v.get(k)]
        country = v.get("country") or v.get("country_code")
        parts = [*cities, country] if country else cities
        parts = [normalize_str(x) for x in parts if normalize_str(x)]
        return " / ".join(parts) if parts else None
    if isinstance(v, (list, tuple)):
        vals = [location_to_str(x) for x in v]
        vals = [x for x in vals if x]
        return " | ".join(vals) if vals else None
    return normalize_str(str(v))

def list_of_locations_to_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, list):
        vals = [location_to_str(x) for x in v]
        vals = [x for x in vals if x]
        return " | ".join(vals) if vals else None
    return location_to_str(v)

def normalize_stage(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).replace("\xa0", " ").strip()
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s

def best_nonnull_column(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    present = [c for c in candidates if c in df.columns]
    if not present:
        return None
    best = None
    best_count = -1
    for c in present:
        cnt = df[c].notna().sum()
        if cnt > best_count:
            best_count = cnt
            best = df[c]
    return best

def clean_nan_str_series(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None:
        return None
    return s.map(lambda x: None if (isinstance(x, str) and x.strip().lower() in {"nan","none","null",""}) else x)

def normalize_tags_value(v: Any) -> Optional[List[str]]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, dict):
                name = normalize_str(x.get("name") or x.get("label") or x.get("tag"))
                if name:
                    out.append(name)
            else:
                sx = normalize_str(str(x))
                if sx:
                    out.append(sx)
        out = [t for t in out if t]
        return list(dict.fromkeys(out)) if out else None
    if isinstance(v, dict):
        name = normalize_str(v.get("name") or v.get("label") or v.get("tag"))
        return [name] if name else None
    s = normalize_str(str(v))
    if not s:
        return None
    parts = [normalize_str(p) for p in re.split(r"[,\|/;]", s) if normalize_str(p)]
    return list(dict.fromkeys(parts)) if parts else None

def normalize_department_value(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, list):
        parts = [normalize_department_value(x) for x in v]
        parts = [p for p in parts if p]
        return " | ".join(parts) if parts else None
    if isinstance(v, dict):
        return normalize_str(v.get("name") or v.get("department") or v.get("team") or v.get("practice"))
    if isinstance(v, str):
        s = v.strip()
        return None if s.lower() in {"nan","none","null",""} else s
    return None

def normalize_reject_reason(v: Any) -> Optional[str]:
    """Normalise la raison de rejet quelle que soit sa forme (dict/list/str)."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, list):
        vals = [normalize_reject_reason(x) for x in v]
        vals = [x for x in vals if x]
        return " | ".join(vals) if vals else None
    if isinstance(v, dict):
        for k in ["label","name","reason","value","title","category","text"]:
            if v.get(k):
                s = str(v.get(k)).strip()
                return None if s.lower() in {"nan","none","null",""} else s
        return None
    s = str(v).strip()
    return None if s.lower() in {"nan","none","null",""} else s

def to_bool_any(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool): 
        return v
    s = str(v).strip().lower()
    if s in {"true","1","yes","y","t"}:
        return True
    if s in {"false","0","no","n","f"}:
        return False
    return None


# ============== Activités (silver applications) ==============

def build_activity_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    1) 'activities_summary' (compteurs + dates) + éventuelles notes textuelles
    2) Fallback 'activities' détaillées
    """
    summary = row.get("activities_summary")
    app_created_iso = to_iso(row.get("created_at") or row.get("application_created_at") or row.get("applied_at"))

    def parse_days_to_first(first_iso: Optional[str]) -> Optional[float]:
        if not first_iso or not app_created_iso:
            return None
        try:
            first = pd.to_datetime(first_iso, utc=True)
            appc = pd.to_datetime(app_created_iso, utc=True)
            val = (first - appc).total_seconds() / 86400.0
            return max(0.0, float(val))
        except Exception:
            return None

    if isinstance(summary, dict) and summary:
        n_notes = int(summary.get("total_notes") or summary.get("notes_count") or 0)
        n_interviews = len(summary.get("interview_kits") or summary.get("interviews") or [])
        n_messages = int(summary.get("messages_count") or summary.get("total_messages") or summary.get("emails_count") or 0)
        n_assess = int(summary.get("assessments_count") or 0)

        first_reply_iso = summary.get("first_reply_at") or summary.get("first_message_at") or None
        last_activity_iso = summary.get("last_activity_at") or None

        first_at = pd.to_datetime(first_reply_iso, utc=True, errors="coerce") if first_reply_iso else None
        last_at = pd.to_datetime(last_activity_iso, utc=True, errors="coerce") if last_activity_iso else None
        appc = pd.to_datetime(app_created_iso, utc=True, errors="coerce") if app_created_iso else None

        days_to_first = parse_days_to_first(first_reply_iso)
        days_in_process = None
        if appc is not None and last_at is not None:
            days_in_process = float((last_at - appc).total_seconds() / 86400.0)
            if days_in_process < 0:
                days_in_process = None

        recent_7d = None
        last_type = None
        if last_at is not None:
            now = datetime.now(timezone.utc)
            recent_7d = (now - last_at.to_pydatetime()).total_seconds() <= 7 * 86400
            last_type = summary.get("last_activity_type")

        notes_lines = []
        msg_lines = []
        assess_lines = []
        interview_lines = []
        if isinstance(summary.get("notes_texts"), list):
            notes_lines = [normalize_str(x) for x in summary["notes_texts"] if normalize_str(x)]
        if isinstance(summary.get("messages_texts"), list):
            msg_lines = [normalize_str(x) for x in summary["messages_texts"] if normalize_str(x)]
        if isinstance(summary.get("assessments_texts"), list):
            assess_lines = [normalize_str(x) for x in summary["assessments_texts"] if normalize_str(x)]
        if isinstance(summary.get("interviews_texts"), list):
            interview_lines = [normalize_str(x) for x in summary["interviews_texts"] if normalize_str(x)]

        return {
            "act_n_total": int(n_notes + n_interviews + n_messages + n_assess),
            "act_n_messages": int(n_messages),
            "act_n_notes": int(n_notes),
            "act_n_interviews": int(n_interviews),
            "act_n_assessments": int(n_assess),
            "act_first_activity_at": first_at.isoformat().replace("+00:00", "Z") if first_at is not None else None,
            "act_last_activity_at": last_at.isoformat().replace("+00:00", "Z") if last_at is not None else None,
            "act_days_to_first_reply": days_to_first,
            "act_days_in_process": days_in_process,
            "act_recent_activity_7d": bool(recent_7d) if recent_7d is not None else None,
            "act_last_activity_type": last_type,
            "text_messages_all": "\n\n---\n\n".join(msg_lines) if msg_lines else None,
            "text_interview_notes_all": "\n\n---\n\n".join(interview_lines) if interview_lines else None,
            "text_assessments_all": "\n\n---\n\n".join(assess_lines) if assess_lines else None,
            "text_last_activities_30d": None,  # non disponible dans le résumé
            "text_notes_all": "\n\n---\n\n".join(notes_lines) if notes_lines else None,
            "y_proxy_positive": True if n_interviews > 0 else False,
            "act_had_interview": True if n_interviews > 0 else False,
        }

    # Fallback: si pas de summary, renvoyer une structure vide pour ne pas bloquer
    return {
        "act_n_total": None,
        "act_n_messages": None,
        "act_n_notes": None,
        "act_n_interviews": None,
        "act_n_assessments": None,
        "act_first_activity_at": None,
        "act_last_activity_at": None,
        "act_days_to_first_reply": None,
        "act_days_in_process": None,
        "act_recent_activity_7d": None,
        "act_last_activity_type": None,
        "text_messages_all": None,
        "text_interview_notes_all": None,
        "text_assessments_all": None,
        "text_last_activities_30d": None,
        "text_notes_all": None,
        "y_proxy_positive": None,
        "act_had_interview": None,
    }


# ============== Segmentation CV (silver candidates) ==============

# ---- Robust helpers for segmentation & dates ----

_MONTH_FR = {
    "janvier":"jan","février":"feb","fevrier":"feb","mars":"mar","avril":"apr","mai":"may","juin":"jun",
    "juillet":"jul","août":"aug","aout":"aug","septembre":"sep","octobre":"oct","novembre":"nov","décembre":"dec","decembre":"dec"
}

def _normalize_months_fr(s: str) -> str:
    t = s.lower()
    for fr, en in _MONTH_FR.items():
        # on applique la substitution sur 't' à chaque itération
        t = re.sub(rf"\b{re.escape(fr)}\b", en, t)
    return t

def parse_date_robust(d):
    if d is None:
        return None
    if isinstance(d, (int, float)):
        try:
            return pd.to_datetime(d, utc=True, errors="coerce")
        except Exception:
            return None
    s = str(d).strip()
    if not s or s.lower() in {"na","n/a","none","null"}:
        return None
    s = (s.replace("Aujourd’hui","Present").replace("Aujourd'hui","Present")
           .replace("Présent","Present").replace("present","Present")
           .replace("Actuel","Present"))
    s_norm = _normalize_months_fr(s)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%m/%Y", "%b %Y", "%B %Y", "%Y"):
        try:
            return pd.to_datetime(s_norm, format=fmt, dayfirst=True, utc=True)
        except Exception:
            pass
    return pd.to_datetime(s_norm, dayfirst=True, utc=True, errors="coerce")

def ensure_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except Exception:
                pass
        return {"description": x}
    return {}

def _normalize_list_of_dicts_or_str(x):
    # returns list of dicts
    if x is None:
        return []
    if isinstance(x, dict):
        return [x]
    if isinstance(x, list):
        return [ensure_dict(t) for t in x]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("["):
            try:
                arr = json.loads(s)
                return [ensure_dict(t) for t in arr]
            except Exception:
                return [ensure_dict(s)]
        return [ensure_dict(s)]
    return [ensure_dict(x)]

def explode_cv_text_views(seg: Dict[str, Any]) -> Tuple[
    Optional[str], Optional[str], Optional[str], Optional[List[str]], Optional[List[str]],
    Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]
]:
    seg = ensure_dict(seg)
    experiences = _normalize_list_of_dicts_or_str(seg.get("experiences") or [])
    education = _normalize_list_of_dicts_or_str(seg.get("education") or [])
    skills = seg.get("skills") or []
    languages = seg.get("languages") or []
    summary = normalize_str(seg.get("summary") or seg.get("about"))

    profile = seg.get("profile")
    p_title = p_loc_raw = None
    if isinstance(profile, dict):
        p_title = normalize_str(profile.get("title_or_role") or profile.get("title"))
        p_loc_raw = profile.get("location")

    # dernière expérience (la plus récente selon date de fin)
    best_e = None
    best_end = None
    for e in experiences:
        end = parse_date_robust(e.get("end_date") or e.get("to") or e.get("end"))
        end_val = end.value if isinstance(end, pd.Timestamp) and pd.notna(end) else float("inf")
        if best_end is None or end_val > best_end:
            best_end = end_val
            best_e = e

    last_title = normalize_str(best_e.get("title") or best_e.get("role")) if isinstance(best_e, dict) else None
    last_company = normalize_str(best_e.get("company") or best_e.get("employer")) if isinstance(best_e, dict) else None
    last_loc = normalize_str(best_e.get("location") or best_e.get("city") or best_e.get("place")) if isinstance(best_e, dict) else None

    # vues texte expériences
    exp_lines = []
    for e in experiences:
        title = normalize_str(e.get("title") or e.get("role"))
        company = normalize_str(e.get("company"))
        start = normalize_str(e.get("start_date") or e.get("from") or e.get("start"))
        end = normalize_str(e.get("end_date") or e.get("to") or e.get("end"))
        loc = normalize_str(e.get("location") or e.get("city") or e.get("place"))
        bullets = e.get("bullets") or []
        desc = normalize_str(e.get("description"))
        if desc:
            bullets = bullets + [desc]
        bullets = [str(b).strip() for b in bullets if str(b).strip() != ""]
        bullets_str = " • ".join(bullets) if bullets else None
        line = " | ".join([x for x in [f"{start or ''}-{end or ''}", title, company, loc, bullets_str] if x])
        if line:
            exp_lines.append(line)
    text_exp = "\n".join(exp_lines) if exp_lines else None

    # vues texte éducation
    edu_lines = []
    for ed in education:
        degree = normalize_str(ed.get("degree"))
        school = normalize_str(ed.get("school"))
        start = normalize_str(ed.get("start_date") or ed.get("from") or ed.get("start"))
        end = normalize_str(ed.get("end_date") or ed.get("to") or ed.get("end"))
        field = normalize_str(ed.get("field") or ed.get("major"))
        line = " | ".join([x for x in [degree, field, school, f"{start or ''}-{end or ''}"] if x])
        if line:
            edu_lines.append(line)
    text_edu = "\n".join(edu_lines) if edu_lines else None

    # langues normalisées
    lang_vals = []
    for l in (languages or []):
        if isinstance(l, dict):
            v = l.get("name") or l.get("language") or l.get("lang")
        else:
            v = str(l)
        if v is not None and str(v).strip() != "":
            lang_vals.append(str(v).strip().lower())
    langs = sorted(list({x for x in lang_vals})) if lang_vals else None
    cv_lang = normalize_str(seg.get("lang") or seg.get("language"))
    if not cv_lang and langs:
        cv_lang = langs[0]

    if not summary and p_title:
        summary = p_title

    clean_skills = []
    if isinstance(skills, list):
        for s in skills:
            if isinstance(s, dict):
                val = s.get("name") or s.get("skill") or ""
            else:
                val = str(s)
            val = (val or "").strip().lower()
            if val:
                clean_skills.append(val)
        clean_skills = list(dict.fromkeys(clean_skills))[:128]

    parts = []
    if text_exp:
        parts.append(text_exp)
    if text_edu:
        parts.append(text_edu)
    if clean_skills:
        parts.append(", ".join(clean_skills[:50]))
    if summary:
        parts.append(summary)
    if p_title:
        parts.append(p_title)
    text_skills = ", ".join(clean_skills[:50]) if clean_skills else None
    text_full = "\n\n---\n\n".join(parts) if parts else None

    return (
        text_exp, text_skills, text_full, (clean_skills or None), langs, summary, cv_lang,
        p_title, location_to_str(p_loc_raw), last_loc, last_title, last_company
    )

def derive_has_segments(seg: Dict[str, Any]) -> bool:
    if not isinstance(seg, dict) or not seg:
        return False
    if seg.get("has_segmented_cv") is True:
        return True
    for k in ["experiences", "skills", "education", "summary", "about"]:
        v = seg.get(k)
        if isinstance(v, list) and len(v) > 0:
            return True
        if isinstance(v, str) and v.strip() != "":
            return True
    profile = seg.get("profile")
    if isinstance(profile, dict):
        if normalize_str(profile.get("title_or_role")) or normalize_str(profile.get("summary")):
            return True
    return False

def map_candidate_seg_fields(df_cand: pd.DataFrame) -> pd.DataFrame:
    df = df_cand.copy()
    if "id" in df.columns and "candidate_id" not in df.columns:
        df = df.rename(columns={"id": "candidate_id"})
    return df

def build_cv_from_candidates_row(row: Dict[str, Any]) -> Dict[str, Any]:
    seg = row.get("cv") or row.get("segments") or row.get("cv_segments") or row.get("segmented_cv") or row.get("cv_json") or {}
    # tolère JSON string ou texte
    if isinstance(seg, str):
        s = seg.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                seg = json.loads(s)
            except Exception:
                seg = {"description": seg}
        else:
            seg = {"description": seg}
    if not isinstance(seg, dict):
        seg = {}

    (text_exp, text_skills, text_full, skills_list, langs_list, summary, cv_lang,
     p_title, p_loc, last_loc, last_title, last_company) = explode_cv_text_views(seg)
    cv_has = derive_has_segments(seg)
    has_segments_top = bool(row.get("has_segments") is True)
    synth_summary = summary
    if not synth_summary:
        parts = [p_title, last_title, last_company]
        parts = [p for p in parts if p]
        synth_summary = " — ".join(parts) if parts else None
    return {
        "cv_lang": cv_lang,
        "cv_has_segments": True if (has_segments_top or cv_has) else False,
        "cv_skills": skills_list,
        "cv_languages": langs_list,
        "cv_summary": synth_summary,
        "text_cv_experience": text_exp,
        "text_cv_skills": text_skills,
        "text_cv_full": text_full,
        "cv_text_hash": row.get("cv_text_hash") or row.get("text_hash") or None,
        "cv_segmenter_version": row.get("segmenter_version") or None,
        "cv_llm_deployment": row.get("llm_deployment") or None,
        "__cv_profile_title": p_title,
        "__cv_profile_location": p_loc,
        "__cv_languages": langs_list,
        "__cv_skills": skills_list,
        "__cv_last_location": last_loc,
        "__cv_last_title": last_title,
        "__cv_last_company": last_company,
    }


# ---- Robust key coalescing for joins ----

def _maybe_json(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except Exception:
                return None
    return None

def _extract_id_from_any(obj: Any, kind: str = "candidate") -> Optional[str]:
    if isinstance(obj, dict):
        for k in ["candidate_id","person_id","id","candidateId","job_id","offer_id","opening_id","jobId"]:
            if k in obj and obj[k]:
                return str(obj[k])
        for k in ["data","candidate","person","applicant","job","opening","offer"]:
            if k in obj and isinstance(obj[k], dict) and "id" in obj[k]:
                return str(obj[k]["id"])
    if isinstance(obj, str):
        if kind == "candidate":
            m = re.search(r"candidates/(\d+)", obj)
            if m: return m.group(1)
        else:
            m = re.search(r"(?:jobs|openings|offers)/(\d+)", obj)
            if m: return m.group(1)
    return None

def coalesce_key(df: pd.DataFrame, target: str, candidates: List[str], kind: str) -> pd.DataFrame:
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            df[target] = df[c]
            return df
    tokens = ["candidate","person","applicant","cv","resume","blob","url","path"] if kind=="candidate" \
             else ["job","offer","opening","posting","url","path","body"]
    guess_cols = [c for c in df.columns if any(tok in c for tok in tokens)]
    for c in guess_cols:
        vals = df[c].map(lambda x: _extract_id_from_any(_maybe_json(x) or x, kind=kind))
        if vals.notna().any():
            df[target] = vals
            return df
    for c in df.columns:
        if df[c].dtype == object:
            ser = df[c].astype(str)
            if kind == "candidate":
                m = ser.str.extract(r"candidates/(\d+)", expand=False)
            else:
                m = ser.str.extract(r"(?:jobs|openings|offers)/(\d+)", expand=False)
            if m.notna().any():
                df[target] = m
                return df
    return df


# ============== Construction GOLD ==============

def construct_gold(
    df_apps: pd.DataFrame,
    df_cand: pd.DataFrame,
    df_jobs: pd.DataFrame,
    only_has_segments: bool = True,
) -> pd.DataFrame:

    # Applications
    apps = df_apps.copy()
    if "id" in apps.columns and "application_id" not in apps.columns:
        apps = apps.rename(columns={"id": "application_id"})
    if "job_id" not in apps.columns:
        for src in ["jobId", "job", "job_id"]:
            if src in apps.columns:
                apps = apps.rename(columns={src: "job_id"})
                break
    if "candidate_id" not in apps.columns:
        for src in ["candidateId", "candidate", "candidate_id", "person_id"]:
            if src in apps.columns:
                apps = apps.rename(columns={src: "candidate_id"})
                break

    # Candidates & Jobs
    cand = map_candidate_seg_fields(df_cand)
    if "id" in cand.columns and "candidate_id" not in cand.columns:
        cand = cand.rename(columns={"id": "candidate_id"})
    jobs = df_jobs.copy()
    if "id" in jobs.columns and "job_id" not in jobs.columns:
        jobs = jobs.rename(columns={"id": "job_id"})

    # Robust coalescing of join keys
    CAND_ID_CANDS = ["candidate_id","person_id","candidateId","candidate.id","applicant_id","applicant.id"]
    JOB_ID_CANDS  = ["job_id","offer_id","opening_id","jobId","job.id","posting_id"]
    apps = coalesce_key(apps, "candidate_id", CAND_ID_CANDS, kind="candidate")
    cand = coalesce_key(cand, "candidate_id", CAND_ID_CANDS, kind="candidate")
    apps = coalesce_key(apps, "job_id", JOB_ID_CANDS, kind="job")
    jobs = coalesce_key(jobs, "job_id", JOB_ID_CANDS, kind="job")

    # Cast ids to string to avoid non-matching merges
    for d, cols in [(apps, ["candidate_id","job_id"]), (cand, ["candidate_id"]), (jobs, ["job_id"])]:
        for c in cols:
            if c in d.columns:
                d[c] = d[c].astype(str)

    # Join
    base = apps.merge(cand, on="candidate_id", how="left", suffixes=("", "_cand"))
    base = base.merge(jobs, on="job_id", how="left", suffixes=("", "_job"))

    # Activities
    feat_records = []
    for _, r in base.iterrows():
        feat_records.append(build_activity_features(r.to_dict()))
    df_feats = pd.DataFrame(feat_records)
    base = pd.concat([base.reset_index(drop=True), df_feats.reset_index(drop=True)], axis=1)

    # CV (depuis silver candidates)
    cv_records = []
    for _, r in cand.iterrows():
        row = r.to_dict()
        cid = row.get("candidate_id")
        rec = {"candidate_id": str(cid) if cid is not None else None}
        rec.update(build_cv_from_candidates_row(row))
        cv_records.append(rec)
    df_cv = pd.DataFrame(cv_records)
    base = base.merge(df_cv, on="candidate_id", how="left")

    # has_segments final : priorité à la colonne explicite si présente, sinon dérivation CV
    if "has_segments" not in base.columns:
        base["has_segments"] = base.get("cv_has_segments")
    else:
        base["has_segments"] = base["has_segments"].fillna(base.get("cv_has_segments"))

    # Filtre optionnel
    if only_has_segments:
        base = base[base["has_segments"] == True].copy()

    # -------- Table finale (mêmes colonnes que l'existant) --------
    out = pd.DataFrame({
        "application_id": base["application_id"].astype(str),
        "job_id": base["job_id"].astype(str),
        "candidate_id": base["candidate_id"].astype(str),
    })

    # Application (dates & stage)
    out["application_created_at"] = pick(base, ["created_at", "application_created_at", "applied_at"]).map(to_iso)
    out["application_stage"] = pick(base, ["stage_name", "status", "stage", "application_stage"]).map(normalize_stage)

    # -------- Job --------
    out["job_title"] = pick(base, ["title_job", "job_title", "title"])

    # Description (scan + html)
    desc_candidates = [
        "job_description_text","description_text","description_job","job_description","description",
        "body_text","body","description_text_job"
    ]
    desc_ser = best_nonnull_column(base, desc_candidates)
    if desc_ser is None:
        dyn_desc_cols = [c for c in base.columns if re.search(r"(desc|body)", c, re.I)]
        desc_ser = best_nonnull_column(base, dyn_desc_cols)
    html_candidates = ["description_html_job","job_description_html","description_html","body_html","body_html_job"]
    html_ser = best_nonnull_column(base, html_candidates)
    if desc_ser is not None and html_ser is not None:
        out["job_description_text"] = desc_ser.combine_first(html_ser.map(html_to_text))
    elif desc_ser is not None:
        out["job_description_text"] = desc_ser
    elif html_ser is not None:
        out["job_description_text"] = html_ser.map(html_to_text)
    else:
        out["job_description_text"] = None

    # Department
    dept_candidates = ["department_job","job_department","department","department_name","dept_job","dept","team","practice"]
    dept_ser = best_nonnull_column(base, dept_candidates)
    if dept_ser is None:
        dyn_dept_cols = [c for c in base.columns if re.search(r"(depart|team|practice)", c, re.I)]
        dept_ser = best_nonnull_column(base, dyn_dept_cols)
    out["job_department"] = (dept_ser.map(normalize_department_value) if dept_ser is not None else None)

    # Location(s)
    loc_candidates = ["location_job","job_location","workplace_job","workplace"]
    loc_ser = best_nonnull_column(base, loc_candidates)
    if loc_ser is None:
        dyn_loc_cols = [c for c in base.columns if re.search(r"(locat|workplace)", c, re.I)]
        loc_ser = best_nonnull_column(base, dyn_loc_cols)
    locs_list_candidates = ["locations_job","job_locations","locations"]
    locs_ser = best_nonnull_column(base, locs_list_candidates)
    if loc_ser is not None and locs_ser is not None:
        jl = loc_ser.combine_first(locs_ser.map(list_of_locations_to_str))
    elif loc_ser is not None:
        jl = loc_ser.map(location_to_str)
    elif locs_ser is not None:
        jl = locs_ser.map(list_of_locations_to_str)
    else:
        jl = None
    if jl is not None:
        jl = jl.map(lambda x: None if (
            x is None or
            (isinstance(x, list) and len(x) == 0) or
            (isinstance(x, str) and x.strip().lower() in {"nan","none","null","[]",""})
        ) else x)
    out["job_location"] = jl

    # Tags
    tags_candidates = ["tags_job","job_tags","tag_list_job","tag_list","labels_job","labels","keywords_job","keywords"]
    tags_ser = best_nonnull_column(base, tags_candidates)
    if tags_ser is None:
        dyn_tag_cols = [c for c in base.columns if re.search(r"(tag|label|keyword)", c, re.I)]
        tags_ser = best_nonnull_column(base, dyn_tag_cols)
    out["job_tags"] = tags_ser.map(normalize_tags_value) if tags_ser is not None else None

    # -------- Candidat --------
    # nom complet si présent, sinon None (ne casse pas le schéma)
    if "first_name" in base.columns or "last_name" in base.columns:
        first = base["first_name"] if "first_name" in base.columns else pd.Series([""]*len(base), index=base.index)
        last = base["last_name"] if "last_name" in base.columns else pd.Series([""]*len(base), index=base.index)
        full_name_series = (first.fillna('') + " " + last.fillna('')).str.strip().replace("", pd.NA)
    else:
        full_name_series = pd.Series([None]*len(base), index=base.index)
    out["cand_full_name"] = full_name_series

    cand_loc_candidates = pick(base, ["candidate_location","location_cand","location"])
    loc_fallback_profile = base.get("__cv_profile_location", pd.Series([None]*len(base)))
    loc_fallback_last = base.get("__cv_last_location", pd.Series([None]*len(base)))
    out["cand_location"] = cand_loc_candidates.combine_first(loc_fallback_profile).combine_first(loc_fallback_last).map(location_to_str)

    out["cand_email_hash"] = pick(base, ["email_hash","candidate_email_hash"])
    out["cand_current_title"] = pick(base, ["current_title","headline","candidate_title"]) \
        .combine_first(base.get("__cv_profile_title", pd.Series([None]*len(base)))) \
        .combine_first(base.get("__cv_last_title", pd.Series([None]*len(base))))
    out["cand_seniority"] = pick(base, ["seniority"])
    out["cand_langs"] = pick(base, ["languages","cand_langs"]).combine_first(base.get("__cv_languages", pd.Series([None]*len(base))))
    out["cand_skills"] = pick(base, ["skills","cand_skills"]).combine_first(base.get("__cv_skills", pd.Series([None]*len(base))))

    # -------- CV (vues & méta) --------
    for c in ["cv_lang","cv_has_segments","cv_skills","cv_languages","cv_summary",
              "text_cv_experience","text_cv_skills","text_cv_full",
              "cv_text_hash","cv_segmenter_version","cv_llm_deployment"]:
        out[c] = base[c] if c in base.columns else None

    # -------- Activités --------
    for c in [
        "act_n_total","act_n_messages","act_n_notes","act_n_interviews","act_n_assessments",
        "act_first_activity_at","act_last_activity_at","act_days_to_first_reply","act_days_in_process",
        "act_recent_activity_7d","act_last_activity_type",
        "text_messages_all","text_interview_notes_all","text_assessments_all","text_last_activities_30d","text_notes_all",
        "act_had_interview","y_proxy_positive"
    ]:
        out[c] = base[c] if c in base.columns else None

    # -------- Outcomes / décisions --------
    out["y_final_decision"] = pick(base, ["final_decision", "decision"])
    out["y_offer_made"] = pick(base, ["offer_made"])
    out["y_hired"] = pick(base, ["hired"])

    # Rejet & embauche: raisons/dates multi-sources
    reject_reason_candidates = [
        "reject_reason_text", "rejection_reason_text",
        "rejection_reason", "reject_reason", "rejected_reason",
        "reject_reason_label","reject_reason_name"
    ]
    out["rejection_reason"] = best_nonnull_column(base, reject_reason_candidates).map(normalize_reject_reason) \
        if best_nonnull_column(base, reject_reason_candidates) is not None else None

    out["rejected_at"] = pick(base, ["rejected_at","rejection_date","rejectedDate"]).map(to_iso)
    out["hired_at"] = pick(base, ["hired_at","offer_accepted_at","offer_signed_at"]).map(to_iso)

    # is_rejected / is_hired
    is_rej = pick(base, ["is_rejected"])
    if is_rej is None or is_rej.notna().sum() == 0:
        is_rej = out["rejected_at"].notna()
    out["is_rejected"] = is_rej.map(lambda x: bool(x) if x is not None else None)

    is_hired = pick(base, ["is_hired"])
    if is_hired is None or is_hired.notna().sum() == 0:
        is_hired = out["hired_at"].notna()
    out["is_hired"] = is_hired.map(lambda x: bool(x) if x is not None else None)

    # application_outcome
    def _outcome(row):
        if row.get("is_hired") is True or str(row.get("y_final_decision")).lower() == "hired":
            return "hired"
        if row.get("is_rejected") is True or str(row.get("y_final_decision")).lower() == "rejected":
            return "rejected"
        return "in_process"
    out["application_outcome"] = out.apply(_outcome, axis=1)

    # Métadonnées
    ts_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    out["meta_processing_ts"] = ts_iso
    out["meta_gold_version"] = "v1.5.3"

    # Dates sûres
    for c in ["act_first_activity_at","act_last_activity_at","application_created_at","rejected_at","hired_at"]:
        if c in out.columns:
            out[c] = out[c].map(to_iso)

    # Dédoublonnage
    out = out.drop_duplicates(subset=["application_id"], keep="last")
    return out


# ============== CLI / main ==============

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build GOLD table from silvers (no external segmentation file).")
    parser.add_argument("--only-has-segments", action="store_true", default=True,
                        help="Garder uniquement les candidatures avec has_segments=True.")
    parser.add_argument("--apps-prefix", default=os.getenv("SILVER_APPLICATIONS_PREFIX", "silver/job-applications/"))
    parser.add_argument("--cands-prefix", default=os.getenv("SILVER_CANDIDATES_PREFIX", "silver/candidates_unified/"))
    parser.add_argument("--jobs-prefix", default=os.getenv("SILVER_JOBS_PREFIX", "silver/jobs/"))
    parser.add_argument("--gold-prefix", default=os.getenv("GOLD_PREFIX", "gold/"))
    parser.add_argument("--container", default=os.getenv("BLOB_CONTAINER", "cvcompat"))
    args = parser.parse_args()

    bsc = make_blob_service_client()
    container_client = bsc.get_container_client(args.container)
    print(f"[gold] Reading silvers from container='{args.container}' ...", file=sys.stderr)
    df_apps = read_latest_jsonl_df(container_client, args.apps_prefix)
    df_cand = read_latest_jsonl_df(container_client, args.cands_prefix)
    df_jobs = read_latest_jsonl_df(container_client, args.jobs_prefix)

    if df_apps.empty:
        raise RuntimeError(f"Aucun fichier d'applications trouvé sous prefix '{args.apps_prefix}'")
    if df_cand.empty:
        print(f"[warn] Aucun candidates sous '{args.cands_prefix}' — filtrage has_segments risque de tout éliminer.", file=sys.stderr)
    if df_jobs.empty:
        print(f"[warn] Aucun jobs sous '{args.jobs_prefix}' — colonnes job_* seront souvent nulles.", file=sys.stderr)

    print("[gold] Constructing GOLD dataframe ...", file=sys.stderr)
    df_gold = construct_gold(df_apps, df_cand, df_jobs, only_has_segments=args.only_has_segments)

    if df_gold.empty:
        print("[gold][warn] GOLD est vide après filtrage. Vérifie la présence de segments côté candidates ou passe --only-has-segments=false", file=sys.stderr)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    base_name = f"applications_gold_{ts}"
    jsonl_name = f"{args.gold_prefix}{base_name}.jsonl"
    parquet_name = f"{args.gold_prefix}{base_name}.parquet"
    latest_jsonl = f"{args.gold_prefix}applications_gold_latest.jsonl"
    latest_parquet = f"{args.gold_prefix}applications_gold_latest.parquet"

    print(f"[gold] Uploading {jsonl_name} & {parquet_name} ...", file=sys.stderr)
    upload_bytes(container_client, jsonl_name, df_to_jsonl_bytes(df_gold), overwrite=True)
    upload_bytes(container_client, parquet_name, df_to_parquet_bytes(df_gold), overwrite=True)
    upload_bytes(container_client, latest_jsonl, df_to_jsonl_bytes(df_gold), overwrite=True)
    upload_bytes(container_client, latest_parquet, df_to_parquet_bytes(df_gold), overwrite=True)

    print(f"[gold] Done. Rows={len(df_gold)} → {jsonl_name} / {parquet_name}", file=sys.stderr)

if __name__ == "__main__":
    main()
