# pages/02_💼_Offres_Publiées.py — Job Offers Visualization
from __future__ import annotations

import json
import os
from datetime import datetime, date
from typing import Optional, Any

import numpy as np
import pandas as pd
import streamlit as st
import requests

# Azure Blob
try:
    from azure.storage.blob import BlobServiceClient  # type: ignore
except Exception:
    BlobServiceClient = None

st.set_page_config(page_title="CV Compatibility — Offres Publiées", page_icon="💼", layout="wide")

# Configuration API
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Liste d'écoles cibles (même liste que dans le notebook ML)
TARGET_ECOLES = [
    # Écoles d'ingénieurs
    "epitech", "ieseg", "em grenoble", "em normandie", "esiea", "eisti", "epita",
    "insa", "esilv", "cy tech", "cytech", "esiee", "em lyon", "edhec", "utc",
    "gobelins", "hetic", "strate", "polytechnique", "polytech", "telecom paris",
    "centrale supelec", "centrale supélec", "mines paris tech", "mines paristech",
    "neoma", "ensimag", "enseeiht", "inp", "kedge", "skema",
    # Universités
    "universite paris-saclay", "universite paris saclay", "paris-saclay", "paris saclay",
    "universite dauphine", "dauphine", "imt",
    # Écoles de design et management
    "ecole de design nantes", "design nantes", "audencia", "ece", "polytech angers", "escp"
]

ENV_AZ = {
    "connection_string": "DefaultEndpointsProtocol=https;AccountName=absaifabrik;AccountKey=dHnt6TqjK6GYHvEjLTKenFdRHitSCByxcqenpCAvP/+GkY6XjHk7+BMfSpVuhbSpUi/5EfGq61CR+AStw0NiCA==;EndpointSuffix=core.windows.net",
    "account_url": os.getenv("AZURE_BLOB_ACCOUNT_URL"),
    "sas_token": os.getenv("AZURE_BLOB_SAS_TOKEN"),
    "account_key": os.getenv("AZURE_STORAGE_ACCOUNT_KEY"),
    "container": os.getenv("AZURE_BLOB_CONTAINER") or "cvcompat",
    "gold_blob": os.getenv("GOLD_BLOB_PATH") or "gold/applications_gold_latest.jsonl",
    "jobs_silver_blob": os.getenv("JOBS_SILVER_BLOB") or "silver/jobs/2025-11-14T14-58-17Z.jsonl",
}

# secrets > env > défauts
CFG = {
    "connection_string":  ENV_AZ["connection_string"],
    "account_url": ENV_AZ["account_url"],
    "sas_token":  ENV_AZ["sas_token"],
    "account_key":  ENV_AZ["account_key"],
    "container":  ENV_AZ["container"],
    "gold_blob":  ENV_AZ["gold_blob"],
    "jobs_silver_blob":  ENV_AZ["jobs_silver_blob"],
}

DATE_COLS = ["application_created_at", "rejected_at", "hired_at", "meta_processing_ts"]

# --- Utils ---
def _get_blob_service_client() -> "BlobServiceClient":
    if BlobServiceClient is None:
        raise RuntimeError("Installe le paquet 'azure-storage-blob'.")
    if CFG["connection_string"]:
        return BlobServiceClient.from_connection_string(CFG["connection_string"])
    if CFG["account_url"] and CFG["sas_token"]:
        return BlobServiceClient(account_url=CFG["account_url"], credential=CFG["sas_token"])
    if CFG["account_url"] and CFG["account_key"]:
        return BlobServiceClient(account_url=CFG["account_url"], credential=CFG["account_key"])
    raise RuntimeError("Config Azure manquante (connection_string ou account_url + sas/account_key).")

def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def _normalize_jsonl_bytes_to_df(data: bytes, limit_rows: Optional[int] = None) -> pd.DataFrame:
    lines = data.splitlines()
    if limit_rows: lines = lines[:limit_rows]
    recs = []
    for ln in lines:
        ln = ln.strip()
        if not ln: continue
        try:
            recs.append(json.loads(ln))
        except Exception:
            continue
    return pd.DataFrame.from_records(recs)

@st.cache_data(ttl=900, show_spinner=True)
def load_gold_from_blob(container: str, blob_path: str, sample_rows: Optional[int] = None) -> pd.DataFrame:
    bsc = _get_blob_service_client()
    blob_client = bsc.get_blob_client(container=container, blob=blob_path)
    try:
        data = blob_client.download_blob(max_concurrency=4).readall()
    except Exception as e:
        raise RuntimeError(f"Impossible de télécharger '{container}/{blob_path}'. Détail: {e}")
    df = _normalize_jsonl_bytes_to_df(data, limit_rows=sample_rows)

    for c in DATE_COLS:
        if c in df.columns:
            df[c] = _safe_to_datetime(df[c])

    if "application_created_at" in df.columns:
        df["application_date"] = df["application_created_at"].dt.date
        df["application_year"] = df["application_created_at"].dt.year

    return df

@st.cache_data(ttl=900, show_spinner=True)
def load_jobs_silver_from_blob(container: str, blob_path: str) -> pd.DataFrame:
    """Charge les jobs silver et détermine le statut d'ouverture."""
    if not blob_path:
        return pd.DataFrame()
    bsc = _get_blob_service_client()
    bc = bsc.get_blob_client(container=container, blob=blob_path)
    try:
        raw = bc.download_blob(max_concurrency=4).readall()
    except Exception:
        return pd.DataFrame()

    txt = raw.decode("utf-8", errors="ignore").strip()
    if "\n" in txt:
        recs = []
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                recs.append(json.loads(ln))
            except Exception:
                pass
        dfm = pd.DataFrame(recs)
    else:
        try:
            obj = json.loads(txt)
            dfm = pd.DataFrame(obj if isinstance(obj, list) else [obj])
        except Exception:
            dfm = pd.DataFrame()

    if dfm.empty:
        return dfm

    # Normalize column names
    if "job_id" not in dfm.columns and "id" in dfm.columns:
        dfm["job_id"] = dfm["id"]
    if "job_title" not in dfm.columns and "title" in dfm.columns:
        dfm["job_title"] = dfm["title"]
    if "job_status" not in dfm.columns and "status" in dfm.columns:
        dfm["job_status"] = dfm["status"]

    if "job_department" not in dfm.columns:
        if "department" in dfm.columns:
            dfm["job_department"] = dfm["department"]
        elif "division" in dfm.columns:
            dfm["job_department"] = dfm["division"]

    if "job_location" not in dfm.columns:
        if "locations" in dfm.columns:
            dfm["job_location"] = dfm["locations"]
        elif "location" in dfm.columns:
            dfm["job_location"] = dfm["location"]

    # Parse dates
    for src, dst in [("created_at","job_created_at"), ("updated_at","job_updated_at"),
                     ("published_at","job_published_at"), ("archived_at","job_archived_at"), ("closed_at","job_closed_at")]:
        if dst not in dfm.columns and src in dfm.columns:
            dfm[dst] = dfm[src]
        if dst in dfm.columns:
            dfm[dst] = pd.to_datetime(dfm[dst], errors="coerce", utc=True)

    # Determine job_open status
    status = dfm.get("job_status", pd.Series(dtype=str)).astype(str).str.lower().str.strip()
    open_terms   = {"open", "active", "published", "live", "visible", "en ligne"}
    closed_terms = {"closed","archived","filled","cancelled","canceled","on hold","on_hold","paused","inactive","draft"}

    is_published = dfm.get("job_published_at", pd.Series(pd.NaT, index=dfm.index)).notna()
    is_archived  = dfm.get("job_archived_at", pd.Series(pd.NaT, index=dfm.index)).notna()
    has_closed   = dfm.get("job_closed_at",   pd.Series(pd.NaT, index=dfm.index)).notna()

    dfm["job_open"] = np.where(
        status.isin(open_terms), True,
        np.where(
            status.isin(closed_terms) | has_closed | is_archived,
            False,
            (is_published & ~is_archived & ~has_closed)
        )
    )

    keep = [
        "job_id", "job_status", "job_open",
        "job_created_at", "job_updated_at",
        "job_published_at", "job_archived_at", "job_closed_at",
        "job_title", "job_department", "job_location",
    ]
    keep = [c for c in keep if c in dfm.columns]
    dfm = dfm[keep].drop_duplicates(subset=["job_id"] if "job_id" in dfm.columns else None)
    return dfm

def _jsonloads_if_json_string(x):
    """Parse JSON string if applicable."""
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x

def _safe_get(d: dict, path: list, default=None) -> Any:
    """Safely navigate nested dict."""
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
        if cur is None:
            return default
    return cur

def _extract_job_segment_info(job_llm_segments: Any) -> dict:
    """Extract key information from job_llm_segments JSON."""
    if not job_llm_segments:
        return {}

    seg = _jsonloads_if_json_string(job_llm_segments)
    if not isinstance(seg, dict):
        return {}

    job = seg.get("job", {})
    sections = job.get("sections", {})

    info = {}

    # Responsibilities
    resp = sections.get("responsibilities", {})
    if isinstance(resp, dict):
        bullets = resp.get("bullets", [])
        info["responsibilities"] = bullets if bullets else []
    else:
        info["responsibilities"] = []

    # Requirements (must-have)
    req_must = sections.get("requirements_must", {})
    if isinstance(req_must, dict):
        info["skills_required"] = req_must.get("skills", [])
        info["requirements_bullets"] = req_must.get("bullets", [])
        years_req = req_must.get("years_required", {})
        if isinstance(years_req, dict):
            info["years_min"] = years_req.get("min", None)
            info["years_preferred"] = years_req.get("preferred", None)
    else:
        info["skills_required"] = []
        info["requirements_bullets"] = []
        info["years_min"] = None
        info["years_preferred"] = None

    # Requirements (nice to have)
    req_nice = sections.get("requirements_nice", {})
    if isinstance(req_nice, dict):
        info["skills_nice"] = req_nice.get("skills", [])
    else:
        info["skills_nice"] = []

    # Education
    edu = sections.get("education", {})
    if isinstance(edu, dict):
        info["education_level"] = edu.get("level", None)
        info["education_fields"] = edu.get("fields", [])
    else:
        info["education_level"] = None
        info["education_fields"] = []

    # Contract & Remote
    contract = sections.get("contract", {})
    if isinstance(contract, dict):
        info["contract_type"] = contract.get("type", None)
    else:
        info["contract_type"] = None

    remote = sections.get("remote", {})
    if isinstance(remote, dict):
        info["remote_type"] = remote.get("type", None)
    else:
        info["remote_type"] = None

    # Salary
    salary = sections.get("salary", {})
    if isinstance(salary, dict):
        info["salary_min"] = salary.get("min", None)
        info["salary_max"] = salary.get("max", None)
        info["salary_currency"] = salary.get("currency", None)
        info["salary_period"] = salary.get("period", None)
    else:
        info["salary_min"] = None
        info["salary_max"] = None
        info["salary_currency"] = None
        info["salary_period"] = None

    # Benefits
    benefits = sections.get("benefits", {})
    if isinstance(benefits, dict):
        info["benefits"] = benefits.get("bullets", [])
    else:
        info["benefits"] = []

    # Company
    company = sections.get("company", {})
    if isinstance(company, dict):
        info["company_name"] = company.get("text", None)
    else:
        info["company_name"] = None

    # Languages
    langs = sections.get("languages", [])
    info["languages"] = langs if isinstance(langs, list) else []

    # Keywords
    info["keywords"] = job.get("keywords", [])

    return info

def _deduplicate_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate jobs, keep most recent."""
    if df.empty or "job_id" not in df.columns:
        return df

    # Sort by application_created_at descending (most recent first)
    if "application_created_at" in df.columns:
        df = df.sort_values("application_created_at", ascending=False, na_position="last")

    # Drop duplicates by job_id, keeping first (most recent)
    return df.drop_duplicates(subset=["job_id"], keep="first").reset_index(drop=True)

def _count_inbox_never_processed(job_id: Any, df_gold: pd.DataFrame) -> int:
    """Count candidates in Inbox who were never processed for a specific job."""
    df_candidates = df_gold[df_gold["job_id"] == job_id].copy()

    df_inbox_never_moved = df_candidates[
        (df_candidates["application_stage"].str.lower().str.strip() == "inbox") &
        (~df_candidates.get("is_rejected", pd.Series(False, index=df_candidates.index))) &
        (~df_candidates.get("is_hired", pd.Series(False, index=df_candidates.index))) &
        (~df_candidates.get("had_positive_stage_but_extracted_late", pd.Series(False, index=df_candidates.index))) &
        (df_candidates.get("act_n_messages", pd.Series(0, index=df_candidates.index)) == 0) &
        (df_candidates.get("act_n_interviews", pd.Series(0, index=df_candidates.index)) == 0) &
        (df_candidates.get("act_n_notes", pd.Series(0, index=df_candidates.index)) == 0)
    ]

    return len(df_inbox_never_moved)

def _call_scoring_api(job_data: dict, cv_data: dict, api_url: str) -> dict:
    """Call the scoring API for a single application."""
    try:
        # Extract relevant job fields
        job_payload = {
            "job_title": job_data.get("job_title", ""),
            "job_description_text": job_data.get("job_description_text", ""),
            "job_requirements_text": job_data.get("job_requirements_text", ""),
            "job_responsibilities_text": job_data.get("job_responsibilities_text", ""),
            "job_required_skills_must": job_data.get("job_required_skills_must", []),
            "job_required_skills_plus": job_data.get("job_required_skills_plus", []),
            "job_languages_required": job_data.get("job_languages_required", []),
        }

        # Extract relevant CV fields
        cv_payload = {
            "text_cv_full": cv_data.get("text_cv_full", ""),
            "text_cv_skills": cv_data.get("text_cv_skills", ""),
            "text_cv_experience": cv_data.get("text_cv_experience", ""),
            "cv_skills": cv_data.get("cv_skills", []),
            "cv_languages": cv_data.get("cv_languages", []),
            "cand_current_title": cv_data.get("cand_current_title", ""),
            "cand_seniority": cv_data.get("cand_seniority", ""),
        }

        response = requests.post(
            f"{api_url}/score",
            json={
                "job": job_payload,
                "cv": cv_payload
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": str(e)}

def _get_score_badge(score: float) -> tuple[str, str]:
    """Return badge text and color based on score."""
    if score >= 0.65:
        return "✅ Bon candidat", "success"
    elif score < 0.30:
        return "❌ Mauvais candidat", "error"
    else:
        return "⚠️ Nécessite expertise RH", "warning"

def _get_teamtailor_url(candidate_id: str) -> str:
    """Generate Teamtailor candidate profile URL."""
    return f"https://app.teamtailor.com/companies/Y60iFkjehVc/candidates/segment/all/candidate/{candidate_id}"

def _get_school_info(cand: pd.Series) -> tuple[str, bool]:
    """
    Extrait l'école du candidat et vérifie si c'est une école cible.

    Returns:
        tuple[str, bool]: (nom_école, is_target_school)
    """
    # D'abord, chercher dans les données structurées (education)
    education = cand.get("cand_education", [])
    if education and isinstance(education, list) and len(education) > 0:
        # Prendre l'école la plus récente (première de la liste)
        first_edu = education[0]
        if isinstance(first_edu, dict):
            school_name = first_edu.get("school", "")
            if school_name:
                # Vérifier si c'est une école cible
                school_lower = school_name.lower()
                is_target = any(ecole in school_lower for ecole in TARGET_ECOLES if ecole)
                return school_name, is_target

    # Sinon, chercher dans le texte complet du CV
    cv_text = cand.get("text_cv_full", "")
    if cv_text and TARGET_ECOLES:
        cv_text_lower = cv_text.lower()
        for ecole in TARGET_ECOLES:
            if ecole and ecole in cv_text_lower:
                # École cible détectée dans le texte
                return ecole.upper(), True

    return "", False

def _render_candidate_card(cand: pd.Series, score: float = None, rank: int = None):
    """Render a single candidate card with detailed information."""
    candidate_id = str(cand.get("candidate_id", ""))
    app_id = str(cand.get("application_id", ""))

    # Create clickable link to Teamtailor
    teamtailor_url = _get_teamtailor_url(candidate_id)

    # Main info row
    col1, col2, col3 = st.columns([3, 2, 1])

    with col1:
        name = cand.get("cand_full_name", "Nom inconnu")

        if rank:
            name_display = f"**{rank}. [{name}]({teamtailor_url})**"
        else:
            name_display = f"**[{name}]({teamtailor_url})**"

        st.markdown(name_display)

        # Score if available
        if score is not None:
            score_pct = f"{score * 100:.1f}%"
            st.caption(f"📊 Score: {score_pct}")

        # School info
        school_name, is_target = _get_school_info(cand)
        if school_name:
            school_display = f"🎓 {school_name}"
            if is_target:
                school_display += " ✨"
            st.caption(school_display)

    with col2:
        title = cand.get("cand_current_title", "")
        if title:
            st.caption(f"💼 {title}")

        # Email
        email = cand.get("cand_email", "")
        if email:
            st.caption(f"📧 {email}")

    with col3:
        # Application date
        app_date = cand.get("application_created_at")
        if app_date and pd.notna(app_date):
            date_str = pd.to_datetime(app_date).strftime("%d/%m/%Y")
            st.caption(f"📅 {date_str}")

    # Additional details in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        # Phone
        phone = cand.get("cand_phone", "")
        if phone:
            st.caption(f"📞 {phone}")

    with col2:
        # Years of experience
        years_exp = cand.get("cv_years_experience_num")
        if years_exp and pd.notna(years_exp):
            st.caption(f"⏳ {int(years_exp)} ans d'expérience")

        # Seniority
        seniority = cand.get("cand_seniority", "")
        if seniority:
            st.caption(f"🎯 Niveau: {seniority}")

    with col3:
        # Languages
        languages = cand.get("cv_languages")
        if languages:
            if isinstance(languages, list):
                langs_str = ", ".join(languages[:3])
                st.caption(f"🗣️ {langs_str}")
            elif isinstance(languages, str):
                st.caption(f"🗣️ {languages}")

    # Education/School (full width row)
    education = cand.get("cv_education") or cand.get("cand_education")
    if education:
        if isinstance(education, list) and education:
            # Afficher les 2 premières écoles (les plus récentes normalement)
            schools_display = []
            for school in education[:2]:
                if isinstance(school, dict):
                    school_name = school.get("school", "")
                    degree = school.get("degree", "")
                    year = school.get("year", "")
                    if school_name:
                        display_text = school_name
                        if degree:
                            display_text += f" - {degree}"
                        if year:
                            display_text += f" ({year})"
                        schools_display.append(display_text)
                elif isinstance(school, str) and school:
                    schools_display.append(school)

            if schools_display:
                st.caption(f"🎓 **Formation**: {' | '.join(schools_display)}")
        elif isinstance(education, str) and education:
            st.caption(f"🎓 **Formation**: {education}")

    # Skills (full width)
    skills = cand.get("cv_skills") or cand.get("cand_skills")
    if skills:
        if isinstance(skills, list):
            skills_str = ", ".join(skills[:8])  # Show first 8 skills
            st.caption(f"🎯 **Compétences**: {skills_str}")
        elif isinstance(skills, str):
            st.caption(f"🎯 **Compétences**: {skills}")

    # Quality scores if available
    spelling_score = cand.get("cv_spelling_score")
    writing_score = cand.get("cv_writing_quality_score")

    if spelling_score or writing_score:
        col1, col2 = st.columns(2)
        with col1:
            if spelling_score and pd.notna(spelling_score):
                st.caption(f"✍️ Orthographe: {spelling_score:.1f}/10")
        with col2:
            if writing_score and pd.notna(writing_score):
                st.caption(f"📝 Qualité rédactionnelle: {writing_score:.1f}/10")

    st.divider()

def _render_job_offer_card(idx: int, row: pd.Series, df_gold: pd.DataFrame, api_url: str):
    """Render a single job offer as an expandable card."""
    job_id = row.get("job_id", f"job_{idx}")
    job_title = row.get("job_title", "Sans titre")
    job_status = row.get("job_status", "")
    job_open = row.get("job_open", False)

    # Extract job details from GOLD
    job_info = _extract_job_segment_info(row.get("job_llm_segments"))

    # Get all candidates who applied to this job
    df_candidates = df_gold[df_gold["job_id"] == job_id].copy()

    # Filter candidates in Inbox who never moved AND were never processed
    # "Never moved and never processed" means:
    #   - application_stage is "Inbox"
    #   - NOT rejected (is_rejected = False)
    #   - NOT hired (is_hired = False)
    #   - No positive stages detected (had_positive_stage_but_extracted_late = False)
    #   - No meaningful activity: no messages, no interviews, no notes
    df_inbox_never_moved = df_candidates[
        (df_candidates["application_stage"].str.lower().str.strip() == "inbox") &
        (~df_candidates.get("is_rejected", pd.Series(False, index=df_candidates.index))) &
        (~df_candidates.get("is_hired", pd.Series(False, index=df_candidates.index))) &
        (~df_candidates.get("had_positive_stage_but_extracted_late", pd.Series(False, index=df_candidates.index))) &
        (df_candidates.get("act_n_messages", pd.Series(0, index=df_candidates.index)) == 0) &
        (df_candidates.get("act_n_interviews", pd.Series(0, index=df_candidates.index)) == 0) &
        (df_candidates.get("act_n_notes", pd.Series(0, index=df_candidates.index)) == 0)
    ]

    # Status badge
    status_badge = "🟢 EN LIGNE" if job_open else "🔴 FERMÉE"

    # Create expander header with title and status
    header = f"**{job_title}** — {status_badge} — 👥 {len(df_candidates)} candidatures ({len(df_inbox_never_moved)} en Inbox)"

    with st.expander(header, expanded=False):
        # Display key metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            if job_status:
                st.metric("Statut", job_status)
            if job_info.get("contract_type"):
                st.metric("Type de Contrat", job_info["contract_type"])
            if job_info.get("remote_type"):
                st.metric("Mode de Travail", job_info["remote_type"])

        with col2:
            if row.get("job_location"):
                st.metric("Localisation", str(row["job_location"]))
            if job_info.get("years_min") is not None:
                years_txt = f"{job_info['years_min']:.0f} ans min"
                if job_info.get("years_preferred") and job_info["years_preferred"] > 0:
                    years_txt += f" ({job_info['years_preferred']:.0f} ans préféré)"
                st.metric("Expérience", years_txt)

        with col3:
            if row.get("job_department"):
                st.metric("Département", str(row["job_department"]))
            if row.get("job_published_at") and pd.notna(row["job_published_at"]):
                pub_date = pd.to_datetime(row["job_published_at"]).strftime("%d/%m/%Y")
                st.metric("Publié le", pub_date)

        st.divider()

        # Salary
        if job_info.get("salary_min") and job_info.get("salary_max"):
            salary_txt = f"{job_info['salary_min']:.0f} - {job_info['salary_max']:.0f} {job_info.get('salary_currency', 'EUR')}"
            period = job_info.get("salary_period", "")
            if period:
                salary_txt += f" / {period}"
            st.subheader("💰 Salaire")
            st.write(salary_txt)
            st.divider()

        # Responsibilities
        if job_info.get("responsibilities"):
            st.subheader("📋 Responsabilités")
            for bullet in job_info["responsibilities"]:
                st.markdown(f"• {bullet}")
            st.divider()

        # Requirements
        col1, col2 = st.columns(2)

        with col1:
            if job_info.get("skills_required"):
                st.subheader("🎯 Compétences Requises")
                for skill in job_info["skills_required"]:
                    st.markdown(f"• {skill}")

            if job_info.get("requirements_bullets"):
                st.subheader("✅ Critères")
                for bullet in job_info["requirements_bullets"]:
                    st.markdown(f"• {bullet}")

        with col2:
            if job_info.get("skills_nice"):
                st.subheader("⭐ Compétences Souhaitées")
                for skill in job_info["skills_nice"]:
                    st.markdown(f"• {skill}")

            if job_info.get("education_level"):
                st.subheader("🎓 Formation")
                st.write(f"**Niveau:** {job_info['education_level']}")
                if job_info.get("education_fields"):
                    st.write("**Domaines:**")
                    for field in job_info["education_fields"]:
                        st.markdown(f"• {field}")

        st.divider()

        # Languages
        if job_info.get("languages"):
            st.subheader("🗣️ Langues Requises")
            lang_list = []
            for lang in job_info["languages"]:
                if isinstance(lang, dict):
                    code = lang.get("code", "?")
                    level = lang.get("level", "?")
                    lang_list.append(f"{code.upper()}: {level}")
                else:
                    lang_list.append(str(lang))
            st.write(", ".join(lang_list))
            st.divider()

        # Keywords
        if job_info.get("keywords"):
            st.subheader("🔑 Mots-clés")
            keywords_str = ", ".join(job_info["keywords"])
            st.caption(keywords_str)
            st.divider()

        # Dates
        col1, col2, col3 = st.columns(3)
        with col1:
            if row.get("job_created_at") and pd.notna(row["job_created_at"]):
                created = pd.to_datetime(row["job_created_at"]).strftime("%d/%m/%Y")
                st.caption(f"📅 Créée: {created}")

        with col2:
            if row.get("job_updated_at") and pd.notna(row["job_updated_at"]):
                updated = pd.to_datetime(row["job_updated_at"]).strftime("%d/%m/%Y")
                st.caption(f"🔄 Mise à jour: {updated}")

        with col3:
            if row.get("job_archived_at") and pd.notna(row["job_archived_at"]):
                archived = pd.to_datetime(row["job_archived_at"]).strftime("%d/%m/%Y")
                st.caption(f"📦 Archivée: {archived}")
            elif row.get("job_closed_at") and pd.notna(row["job_closed_at"]):
                closed = pd.to_datetime(row["job_closed_at"]).strftime("%d/%m/%Y")
                st.caption(f"❌ Fermée: {closed}")

        st.divider()

        # Candidates section
        st.subheader(f"👥 Candidatures ({len(df_candidates)} total)")

        # Show inbox candidates in an expander
        with st.expander(f"📥 Candidats en Inbox JAMAIS traités ({len(df_inbox_never_moved)})", expanded=False):
            if df_inbox_never_moved.empty:
                st.info("✅ Aucun candidat en Inbox sans traitement (tous ont été évalués ou déplacés).")
            else:
                # Button to score all candidates
                score_key = f"score_btn_{job_id}"
                if st.button(f"🤖 Scorer tous les candidats ({len(df_inbox_never_moved)})", key=score_key):
                    # Initialize session state for scores if not exists
                    if f"scores_{job_id}" not in st.session_state:
                        st.session_state[f"scores_{job_id}"] = {}

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Get job data once (same for all candidates)
                    job_data = row.to_dict()

                    for i, (_, cand) in enumerate(df_inbox_never_moved.iterrows()):
                        app_id = str(cand.get("application_id"))
                        status_text.text(f"Scoring {i+1}/{len(df_inbox_never_moved)}: {cand.get('cand_full_name', 'Unknown')}")

                        # Prepare CV data from candidate
                        cv_data = cand.to_dict()

                        result = _call_scoring_api(job_data, cv_data, api_url)
                        st.session_state[f"scores_{job_id}"][app_id] = result

                        progress_bar.progress((i + 1) / len(df_inbox_never_moved))

                    status_text.text("✅ Scoring terminé!")
                    progress_bar.empty()
                    st.rerun()

                st.divider()

                # Check if we have scores for this job
                has_scores = f"scores_{job_id}" in st.session_state and st.session_state[f"scores_{job_id}"]

                if has_scores:
                    # Categorize candidates by score into 3 groups
                    good_candidates = []  # >= 65%
                    medium_candidates = []  # 30% - 65%
                    bad_candidates = []  # < 30%

                    for _, cand in df_inbox_never_moved.iterrows():
                        app_id = str(cand.get("application_id"))
                        score_data = st.session_state[f"scores_{job_id}"].get(app_id)

                        if score_data and "error" not in score_data:
                            score = score_data.get("score_final", score_data.get("score", 0))
                            cand_info = {
                                "cand": cand,
                                "score": score,
                                "app_id": app_id
                            }

                            if score >= 0.65:
                                good_candidates.append(cand_info)
                            elif score < 0.30:
                                bad_candidates.append(cand_info)
                            else:
                                medium_candidates.append(cand_info)

                    # Sort each group by score (descending)
                    good_candidates.sort(key=lambda x: x["score"], reverse=True)
                    medium_candidates.sort(key=lambda x: x["score"], reverse=True)
                    bad_candidates.sort(key=lambda x: x["score"], reverse=True)

                    # Display Good Candidates Block
                    st.success(f"✅ **Bons candidats** ({len(good_candidates)})")
                    if good_candidates:
                        for i, cand_info in enumerate(good_candidates, 1):
                            _render_candidate_card(cand_info["cand"], score=cand_info["score"], rank=i)
                    else:
                        st.caption("Aucun candidat dans cette catégorie")

                    # Display Medium Candidates Block
                    st.warning(f"⚠️ **Nécessitent expertise RH** ({len(medium_candidates)})")
                    if medium_candidates:
                        for i, cand_info in enumerate(medium_candidates, 1):
                            _render_candidate_card(cand_info["cand"], score=cand_info["score"], rank=i)
                    else:
                        st.caption("Aucun candidat dans cette catégorie")

                    # Display Bad Candidates Block
                    st.error(f"❌ **Mauvais candidats** ({len(bad_candidates)})")
                    if bad_candidates:
                        for i, cand_info in enumerate(bad_candidates, 1):
                            _render_candidate_card(cand_info["cand"], score=cand_info["score"], rank=i)
                    else:
                        st.caption("Aucun candidat dans cette catégorie")

                else:
                    # If no scores yet, display default list
                    for i, (_, cand) in enumerate(df_inbox_never_moved.iterrows(), 1):
                        _render_candidate_card(cand, score=None, rank=i)

        # Show all other candidates
        df_other_candidates = df_candidates[~df_candidates.index.isin(df_inbox_never_moved.index)]

        if not df_other_candidates.empty:
            with st.expander(f"📊 Autres candidatures ({len(df_other_candidates)})", expanded=False):
                # Group by stage
                stage_counts = df_other_candidates["application_stage"].value_counts()

                st.write("**Répartition par statut:**")
                for stage, count in stage_counts.items():
                    st.write(f"• {stage}: {count}")

        st.divider()

        # Footer: Job ID
        st.caption(f"🆔 Job ID: {job_id}")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Paramètres & Source")
    st.caption("SILVER (filtrage) + GOLD (contenu)")
    st.text_input("Container", value=CFG["container"], key="container")
    st.text_input("Blob GOLD (applications)", value=CFG["gold_blob"], key="gold_blob")
    st.text_input("Blob SILVER (jobs)", value=CFG["jobs_silver_blob"], key="jobs_silver_blob")
    st.divider()
    st.subheader("🤖 API Scoring")
    api_url = st.text_input("URL API", value=API_URL, key="api_url")
    st.divider()
    show_only_open = st.toggle("Afficher uniquement les offres EN LIGNE", value=True)
    show_only_with_inbox = st.toggle("Afficher uniquement les offres avec candidats à traiter", value=False)
    st.divider()
    with st.expander("Config détectée"):
        st.code(
            {
                "has_connection_string": bool(CFG["connection_string"]),
                "has_account_url": bool(CFG["account_url"]),
                "has_sas_token": bool(CFG["sas_token"]),
                "has_account_key": bool(CFG["account_key"]),
                "container": CFG["container"],
                "gold_blob": CFG["gold_blob"],
                "jobs_silver_blob": CFG["jobs_silver_blob"],
            },
            language="json",
        )

# --- Load data ---
st.title("💼 Offres Publiées — Visualisation")
st.caption("Source: SILVER (filtrage statut) + GOLD (contenu enrichi)")

if BlobServiceClient is None:
    st.error("Le paquet `azure-storage-blob` est requis. `pip install azure-storage-blob`")
    st.stop()

# Load SILVER to get open job IDs
try:
    df_silver = load_jobs_silver_from_blob(st.session_state["container"], st.session_state["jobs_silver_blob"])
except Exception as e:
    st.error(f"Erreur chargement SILVER: {e}")
    st.stop()

if df_silver.empty:
    st.warning("Aucune offre trouvée dans SILVER."); st.stop()

# Filter for open jobs
if show_only_open:
    df_silver_filtered = df_silver[df_silver["job_open"] == True].reset_index(drop=True)
else:
    df_silver_filtered = df_silver.copy()

# Get list of open job IDs
open_job_ids = set(df_silver_filtered["job_id"].astype(str))

st.write(f"📊 **{len(open_job_ids)}** offres {'EN LIGNE' if show_only_open else 'trouvées'}")

# Load GOLD to get full job content
try:
    df_gold = load_gold_from_blob(st.session_state["container"], st.session_state["gold_blob"])
except Exception as e:
    st.error(f"Erreur chargement GOLD: {e}")
    st.stop()

if df_gold.empty:
    st.warning("Le GOLD est vide."); st.stop()

# Filter GOLD to only jobs that are open (based on SILVER)
df_gold["job_id_str"] = df_gold["job_id"].astype(str)
df_gold_open = df_gold[df_gold["job_id_str"].isin(open_job_ids)].copy()

# Deduplicate by job_id (keep most recent)
df_jobs = _deduplicate_jobs(df_gold_open)

# Merge with SILVER to get status info (only merge columns that exist in SILVER)
silver_cols_to_merge = ["job_id", "job_status", "job_open"]
optional_silver_cols = ["job_published_at", "job_created_at", "job_updated_at", "job_archived_at", "job_closed_at"]

for col in optional_silver_cols:
    if col in df_silver_filtered.columns:
        silver_cols_to_merge.append(col)

df_jobs = df_jobs.merge(
    df_silver_filtered[silver_cols_to_merge],
    on="job_id",
    how="left",
    suffixes=("", "_silver")
)

status_text = "EN LIGNE" if show_only_open else "Tous les statuts"
st.write(f"📊 **{len(df_jobs)}** offres avec contenu ({status_text})")

# Apply inbox candidates filter
if show_only_with_inbox:
    # Calculate inbox count for each job
    df_jobs["inbox_count"] = df_jobs["job_id"].apply(lambda jid: _count_inbox_never_processed(jid, df_gold))

    # Filter to only jobs with inbox candidates
    df_jobs = df_jobs[df_jobs["inbox_count"] > 0].copy()
    st.info(f"🔍 **{len(df_jobs)}** offres avec candidats à traiter")

# --- Filters ---
st.divider()

search_title = st.text_input("🔍 Chercher par titre", placeholder="ex: Data Engineer")

# Apply filters
df_filtered = df_jobs.copy()

if search_title:
    df_filtered = df_filtered[
        df_filtered["job_title"].str.contains(search_title, case=False, na=False)
    ]


# --- Display job offers ---
st.divider()

if df_filtered.empty:
    st.warning("Aucune offre ne correspond aux filtres sélectionnés.")
else:
    for idx, (_, row) in enumerate(df_filtered.iterrows()):
        _render_job_offer_card(idx, row, df_gold, st.session_state["api_url"])
