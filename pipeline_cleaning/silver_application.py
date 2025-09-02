#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, re, math, datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.exceptions import ResourceNotFoundError

# =========================
# Config (override par .env)
# =========================
CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER", "cvcompat")
CONN_STR = "DefaultEndpointsProtocol=https;AccountName=absaifabrik;AccountKey=dHnt6TqjK6GYHvEjLTKenFdRHitSCByxcqenpCAvP/+GkY6XjHk7+BMfSpVuhbSpUi/5EfGq61CR+AStw0NiCA==;EndpointSuffix=core.windows.net"


BRONZE_JOBS_PREFIX        = os.getenv("BRONZE_JOBS_PREFIX", "bronze/jobs/")
BRONZE_APPS_PREFIX        = os.getenv("BRONZE_JOB_APPS_PREFIX", "bronze/job-applications/")
BRONZE_CANDIDATES_PREFIX  = os.getenv("BRONZE_CANDIDATES_PREFIX", "bronze/candidates/")

SILVER_APPS_DIR           = os.getenv("SILVER_JOB_APPS_PREFIX", "silver/job-applications/")
SILVER_OUT_BASENAME       = os.getenv("SILVER_JOB_APPS_BASENAME", "job_applications_enriched.jsonl")
SILVER_OUT_PATH           = f"{SILVER_APPS_DIR}{SILVER_OUT_BASENAME}"
MANIFEST_PATH             = f"{SILVER_OUT_PATH.replace('.jsonl', '_manifest.json')}"

# =========================
# Utils
# =========================
def utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def to_iso(x: Optional[str]) -> Optional[str]:
    return x if x else None

def parse_dt(x: Optional[str]) -> Optional[dt.datetime]:
    if not x: return None
    try:
        return dt.datetime.fromisoformat(x.replace("Z", "+00:00"))
    except Exception:
        return None

def days_between(a: Optional[str], b: Optional[str]) -> Optional[float]:
    da, db = parse_dt(a), parse_dt(b)
    if not (da and db): return None
    return round((db - da).total_seconds() / 86400.0, 6)

def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)

def jloads_safe(s: Optional[str]) -> Dict[str, Any]:
    if not s: return {}
    try:
        return json.loads(s)
    except Exception:
        return {}

def get_env_conn() -> str:
    global CONN_STR
    if not CONN_STR:
        load_dotenv()
        CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not CONN_STR:
        print("❌ AZURE_STORAGE_CONNECTION_STRING manquante", file=sys.stderr)
        sys.exit(1)
    return CONN_STR

# =========================
# Azure I/O
# =========================
def get_container():
    conn = get_env_conn()
    bsc = BlobServiceClient.from_connection_string(conn)
    cc = bsc.get_container_client(CONTAINER_NAME)
    try:
        cc.get_container_properties()
    except ResourceNotFoundError:
        cc.create_container()
    return cc

def iter_blob_texts(cc, prefix: str):
    for b in cc.list_blobs(name_starts_with=prefix):
        if not b.size: 
            continue
        text = cc.get_blob_client(b.name).download_blob().readall().decode("utf-8")
        yield b.name, text

def blob_exists(cc, path: str) -> bool:
    try:
        cc.get_blob_client(path).get_blob_properties()
        return True
    except Exception:
        return False

def upload_text(cc, path: str, text: str):
    cc.get_blob_client(path).upload_blob(
        text.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json; charset=utf-8")
    )

# =========================
# Jobs lookup (léger, aplati)
# =========================
def _extract_one_name(included: List[Dict[str, Any]], type_name: str) -> Optional[str]:
    for inc in included or []:
        if inc.get("type") == type_name:
            ia = inc.get("attributes") or {}
            return ia.get("name") or ia.get("title")
    return None

def norm_status(x: Optional[str]) -> Optional[str]:
    return (x or "").strip().lower() or None

def norm_employment_type(x: Optional[str]) -> Optional[str]:
    return (x or "").strip().lower() or None

def norm_employment_level(x: Optional[str]) -> Optional[str]:
    return (x or "").strip().lower() or None

def norm_remote_status(x: Optional[str]) -> Optional[str]:
    return (x or "").strip().lower() or None

def build_job_lookup(cc, jobs_prefix: str) -> Dict[str, Dict[str, Any]]:
    lut: Dict[str, Dict[str, Any]] = {}
    for _, txt in iter_blob_texts(cc, jobs_prefix):
        try:
            doc = json.loads(txt)
        except Exception:
            continue
        data = doc.get("data") or {}
        if data.get("type") != "jobs":
            continue
        job_id = str(data.get("id"))
        a = data.get("attributes") or {}
        inc = doc.get("included") or []

        # tags (si inclus dans included -> type "tags" / "job-tags" selon export)
        tags = []
        for i in inc:
            if i.get("type") in {"tags", "job-tags"}:
                ia = i.get("attributes") or {}
                t = ia.get("name") or ia.get("title")
                if t: tags.append(t)

        # locations (si inclus)
        locs = []
        for i in inc:
            if i.get("type") in {"locations", "location"}:
                ia = i.get("attributes") or {}
                city = ia.get("city") or ia.get("name")
                country = ia.get("country") or ia.get("country-code")
                locs.append({"city": city, "country": country})

        lut[job_id] = {
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
            "job_locations": locs or None,
            "job_department": _extract_one_name(inc, "departments") or _extract_one_name(inc, "department"),
            "job_division": _extract_one_name(inc, "divisions") or _extract_one_name(inc, "division"),
            "job_recruiter_email": a.get("recruiter-email"),
            "job_tags": tags or None,
        }
    return lut

# =========================
# Candidate enrichment (activities + Q&A)
# =========================
def norm_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80]

def parse_candidate_enrichment(cand_doc: Dict[str, Any]) -> Dict[str, Any]:
    inc = cand_doc.get("included") or []

    activities = [x for x in inc if x.get("type") == "activities"]
    answers    = [x for x in inc if x.get("type") == "answers"]
    questions  = [x for x in inc if x.get("type") == "questions"]

    # --- Map question id -> title
    q_title: Dict[str, str] = {}
    for q in questions:
        qid = str(q.get("id"))
        qa  = q.get("attributes") or {}
        title = (qa.get("title") or "").strip()
        if qid and title:
            q_title[qid] = title  # e.g. "Quelles sont vos attentes salariales ?" etc.  # :contentReference[oaicite:3]{index=3}

    # --- Q&A pivot (souple)
    qa_raw = []
    for a in answers:
        aa = a.get("attributes") or {}
        rel = a.get("relationships") or {}
        qid = None
        if "question" in rel and "links" in rel["question"]:
            # offline: pas de follow; on ne récupère pas l'id facilement → on restera tolérant
            pass
        qa_raw.append({
            "answer_id": str(a.get("id")),
            "question_title": None,  # on peut tenter d’enrichir plus tard si tu as un mapping id
            "text": aa.get("text") or aa.get("answer") or None,
            "question_type": aa.get("question-type"),
            "created_at": to_iso(aa.get("created-at")),
            "updated_at": to_iso(aa.get("updated-at")),
        })
    # Heuristique simple : extraire des champs usuels si on reconnait des titres (quand présents)
    def pick_contains(substr: str) -> Optional[str]:
        for r in qa_raw:
            t = (r.get("question_title") or "").lower()
            if substr.lower() in t and r.get("text"):
                return r["text"]
        return None

    qa_summary = {
        "raw": qa_raw,
        "expectations_salary": pick_contains("attentes salariales"),  # :contentReference[oaicite:4]{index=4}
        "availability":        pick_contains("disponibilit"),         # :contentReference[oaicite:5]{index=5}
        "years_experience":    pick_contains("années d'expérience"),  # :contentReference[oaicite:6]{index=6}
        "managed_team":        pick_contains("managé une équipe"),    # :contentReference[oaicite:7]{index=7}
        "clients_network":     pick_contains("réseau clients"),       # :contentReference[oaicite:8]{index=8}
    }

    # --- Activities aggregation
    activities_sorted = sorted(activities, key=lambda x: (x.get("attributes", {}).get("created-at") or ""))

    stage_events = []
    notes = []
    interview_kits = []
    ratings = []
    tags_current = []
    consent_requested_at = None
    consent_missing_flag = False
    last_activity_at = None
    recruiter_touch_count = 0

    current_stage = None
    last_stage_change_at = None
    stage_durations_days: Dict[str, float] = defaultdict(float)

    for act in activities_sorted:
        at = act.get("attributes") or {}
        code = (at.get("code") or "").strip()
        created_at = to_iso(at.get("created-at"))
        data = jloads_safe(at.get("data"))
        last_activity_at = created_at or last_activity_at

        if code not in {"copilot_resume_summary", "sourced"}:
            recruiter_touch_count += 1  # mesure simple du "touch" recruteur

        if code == "stage":
            f = data.get("from"); t = data.get("to")  # ex. {"from": 39922334, "to": 39922335}  # :contentReference[oaicite:9]{index=9}
            stage_events.append((created_at, f, t))
            if current_stage is not None and last_stage_change_at:
                d = days_between(last_stage_change_at, created_at) or 0.0
                stage_durations_days[str(current_stage)] += d
            current_stage = t
            last_stage_change_at = created_at

        elif code == "note":
            txt = data.get("note") or ""
            notes.append({"at": created_at, "text": txt})  # ex. CR long "les 3 fantastiques…"  # :contentReference[oaicite:10]{index=10}

        elif code == "interview_added":
            kit = data.get("interview_kit_name")  # ex. "Business Plan - Manager et +"  # :contentReference[oaicite:11]{index=11}
            if kit: interview_kits.append(kit)

        elif code == "review":
            try:
                r = float(data.get("rating") or 0)  # ex. {"rating":5}  # :contentReference[oaicite:12]{index=12}
                if r > 0: ratings.append(r)
            except Exception:
                pass

        elif code == "tags":
            to_ = data.get("to")
            if isinstance(to_, list):
                for t in to_:
                    if isinstance(t, str):
                        tags_current.append(t.strip())  # ex. "cabinet upward"  # :contentReference[oaicite:13]{index=13}
                    elif isinstance(t, dict) and t.get("name"):
                        tags_current.append(str(t["name"]).strip())  # ex. "UPWARD"  # :contentReference[oaicite:14]{index=14}

        elif code == "consent_requested":
            consent_requested_at = created_at  # :contentReference[oaicite:15]{index=15}

        elif code == "consent_missing":
            consent_missing_flag = True        # :contentReference[oaicite:16]{index=16}

    # fermer la dernière étape vers "maintenant" (optionnel)
    if current_stage is not None and last_stage_change_at:
        d = days_between(last_stage_change_at, utc_iso()) or 0.0
        stage_durations_days[str(current_stage)] += max(0.0, d)

    notes_sorted = sorted(notes, key=lambda x: x["at"] or "")
    last_note_at = notes_sorted[-1]["at"] if notes_sorted else None
    last_note_excerpt = None
    if notes_sorted:
        t = (notes_sorted[-1]["text"] or "").strip()
        last_note_excerpt = (t[:500] + "…") if len(t) > 500 else t

    first_stage_id = stage_events[0][2] if stage_events else None
    last_stage_id  = stage_events[-1][2] if stage_events else None
    stage_transitions_count = len(stage_events)
    days_in_pipeline = days_between(stage_events[0][0], stage_events[-1][0]) if stage_events else None

    activities_summary = {
        "first_stage_id": first_stage_id,
        "last_stage_id": last_stage_id,
        "stage_transitions_count": stage_transitions_count,
        "stage_durations_days": dict(stage_durations_days),
        "days_in_pipeline": days_in_pipeline,
        "interview_kits": sorted(set(interview_kits)),
        "last_interview_at": None,  # ajoutable si tu veux suivre le max created-at des interview_added
        "total_notes": len(notes_sorted),
        "last_note_at": last_note_at,
        "last_note_excerpt": last_note_excerpt,
        "rating_max": max(ratings) if ratings else None,
        "rating_mean": round(sum(ratings)/len(ratings), 3) if ratings else None,
        "tags_current": sorted(set([t for t in tags_current if t])),
        "consent_requested_at": consent_requested_at,
        "consent_missing_flag": consent_missing_flag,
        "recruiter_touch_count": recruiter_touch_count,
        "last_activity_at": last_activity_at,
    }

    return {
        "activities_summary": activities_summary,
        "qa_summary": qa_summary
    }

# =========================
# Build application row (à partir du bronze job-application)
# =========================
def safe_get(d: Dict[str, Any], path: List[str]) -> Any:
    cur = d
    for k in path:
        cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is None: return None
    return cur

def build_application_row(app_doc: Dict[str, Any], job_lut: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    data = app_doc.get("data") or {}
    if data.get("type") != "job-applications":
        return None

    a   = data.get("attributes") or {}
    rel = data.get("relationships") or {}
    inc = app_doc.get("included") or []

    candidate_id     = safe_get(rel, ["candidate", "data", "id"])
    job_id           = safe_get(rel, ["job", "data", "id"])
    stage_id         = safe_get(rel, ["stage", "data", "id"])
    reject_reason_id = safe_get(rel, ["reject-reason", "data", "id"])

    stage_name = None
    reject_reason_text = None
    for i in inc:
        if i.get("type") == "stages" and i.get("id") == stage_id:
            stage_name = safe_get(i, ["attributes", "name"])
        if i.get("type") == "reject-reasons" and i.get("id") == reject_reason_id:
            reject_reason_text = (
                safe_get(i, ["attributes", "reason"]) or
                safe_get(i, ["attributes", "name"])   or
                safe_get(i, ["attributes", "title"])  or
                safe_get(i, ["attributes", "text"])
            )

    if not reject_reason_text:
        reject_reason_text = (
            a.get("reject-reason-text") or
            a.get("reject_reason_text") or
            a.get("reject-reason")
        )

    created_at      = to_iso(a.get("created-at"))
    updated_at      = to_iso(a.get("updated-at"))
    changed_stage_at= to_iso(a.get("changed-stage-at"))
    rejected_at     = to_iso(a.get("rejected-at"))

    decision = "rejected" if rejected_at else "in_process"

    row: Dict[str, Any] = {
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

    # Snapshot job aplati
    if job_id and str(job_id) in job_lut:
        row.update(job_lut[str(job_id)])

    return row

# =========================
# Main pipeline
# =========================
def main():
    cc = get_container()

    # 1) Lookup jobs
    print("🧭 Indexation jobs (bronze)…")
    job_lut = build_job_lookup(cc, BRONZE_JOBS_PREFIX)
    print(f"   → {len(job_lut)} jobs indexés")

    out_lines: List[str] = []
    apps_total = 0
    apps_built = 0
    apps_enriched = 0
    candidates_missing = 0

    # 2) Parcours de toutes les job-applications (bronze)
    print("🧭 Lecture job-applications (bronze)…")
    for app_path, app_text in iter_blob_texts(cc, BRONZE_APPS_PREFIX):
        try:
            app_doc = json.loads(app_text)
        except Exception:
            continue

        row = build_application_row(app_doc, job_lut)
        if not row:
            continue
        apps_total += 1

        candidate_id = row.get("candidate_id")
        cand_blob = f"{BRONZE_CANDIDATES_PREFIX}{candidate_id}.json" if candidate_id else None

        # 3) Enrichissement activities + Q&A depuis le bronze candidate
        if candidate_id and cand_blob and blob_exists(cc, cand_blob):
            try:
                cand_text = cc.get_blob_client(cand_blob).download_blob().readall().decode("utf-8")
                cand_doc = json.loads(cand_text)
                enrich = parse_candidate_enrichment(cand_doc)
                row["activities_summary"] = enrich.get("activities_summary")
                row["qa_summary"] = enrich.get("qa_summary")
                apps_enriched += 1
            except Exception:
                row["activities_summary"] = None
                row["qa_summary"] = None
        else:
            candidates_missing += 1
            row["activities_summary"] = None
            row["qa_summary"] = None

        out_lines.append(jdump(row))
        apps_built += 1

    # 4) Écriture du silver unique
    if not out_lines:
        print("⚠️ Aucune job-application construite. Vérifie les préfixes BRONZE_*", file=sys.stderr)

    upload_text(cc, SILVER_OUT_PATH, "\n".join(out_lines))
    print(f"✅ Silver écrit → {SILVER_OUT_PATH}")

    # 5) Manifest
    manifest = {
        "generated_at": utc_iso(),
        "inputs": {
            "bronze_jobs_prefix": BRONZE_JOBS_PREFIX,
            "bronze_job_applications_prefix": BRONZE_APPS_PREFIX,
            "bronze_candidates_prefix": BRONZE_CANDIDATES_PREFIX,
        },
        "output": SILVER_OUT_PATH,
        "coverage": {
            "applications_seen": apps_total,
            "applications_built": apps_built,
            "applications_enriched": apps_enriched,
            "applications_without_candidate_json": candidates_missing,
            "enrichment_rate_pct": round(100.0*apps_enriched / apps_built, 2) if apps_built else 0.0
        }
    }
    upload_text(cc, MANIFEST_PATH, jdump(manifest))
    print(f"🧾 Manifest → {MANIFEST_PATH}")

if __name__ == "__main__":
    main()
