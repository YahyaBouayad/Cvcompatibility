# pages/01_📋_Jobs_et_Candidats.py
from __future__ import annotations

import json
import os
from datetime import datetime, date
from typing import Optional, Any, Iterable

import numpy as np
import pandas as pd
import streamlit as st

# Azure Blob
try:
    from azure.storage.blob import BlobServiceClient  # type: ignore
except Exception:
    BlobServiceClient = None

st.set_page_config(page_title="Jobs publiés & candidats — vue cartes", page_icon="📋", layout="wide")

# ───────────────────────────── Config (secrets > env > défauts)
ENV_AZ = {
    "connection_string": "DefaultEndpointsProtocol=https;AccountName=absaifabrik;AccountKey=dHnt6TqjK6GYHvEjLTKenFdRHitSCByxcqenpCAvP/+GkY6XjHk7+BMfSpVuhbSpUi/5EfGq61CR+AStw0NiCA==;EndpointSuffix=core.windows.net",
    "account_url": os.getenv("AZURE_BLOB_ACCOUNT_URL") or "",
    "sas_token": os.getenv("AZURE_BLOB_SAS_TOKEN") or "",
    "account_key": os.getenv("AZURE_STORAGE_ACCOUNT_KEY") or "",
    "container": os.getenv("AZURE_BLOB_CONTAINER") or "cvcompat",
    "gold_blob": os.getenv("GOLD_BLOB_PATH") or "gold/applications_gold_latest.jsonl",
    "jobs_silver_blob": os.getenv("JOBS_SILVER_BLOB") or "silver/jobs/2025-09-12T12-22-00Z.jsonl",  # <- par défaut
}
CFG = {
    "connection_string": ENV_AZ["connection_string"],
    "account_url":  ENV_AZ["account_url"],
    "sas_token":  ENV_AZ["sas_token"],
    "account_key":  ENV_AZ["account_key"],
    "container":  ENV_AZ["container"],
    "gold_blob":  ENV_AZ["gold_blob"],
    "jobs_silver_blob":  ENV_AZ["jobs_silver_blob"],
}

DATE_COLS = ["application_created_at", "rejected_at", "hired_at", "meta_processing_ts"]

# ───────────────────────────── Utils chargement & normalisation
def _get_blob_service_client() -> "BlobServiceClient":
    if BlobServiceClient is None:
        raise RuntimeError("Le paquet 'azure-storage-blob' est requis. `pip install azure-storage-blob`")
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
        if not ln: 
            continue
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

    if "application_outcome" in df.columns:
        df["application_outcome"] = df["application_outcome"].fillna("unknown")

    for b in ("is_rejected", "y_hired", "y_offer_made", "act_had_interview"):
        if b in df.columns:
            df[b] = df[b].fillna(False).astype(bool)

    return df

# Normalise "locations" → string
def _first_location_to_str(v):
    if v is None:
        return ""
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                v = json.loads(s)
            except Exception:
                return s
        else:
            return s
    if isinstance(v, list) and v:
        first = v[0]
        if isinstance(first, dict):
            for k in ("name","city","display","label"):
                if k in first and first[k]:
                    return str(first[k])
            return str(first)
        return str(first)
    if isinstance(v, dict):
        for k in ("name","city","display","label"):
            if k in v and v[k]:
                return str(v[k])
        return str(v)
    return str(v)

# SILVER Jobs loader avec statut robuste & coalesce des champs
@st.cache_data(ttl=900, show_spinner=True)
def load_jobs_silver_from_blob(container: str, blob_path: str) -> pd.DataFrame:
    if not blob_path:
        return pd.DataFrame()
    bsc = _get_blob_service_client()
    bc = bsc.get_blob_client(container=container, blob=blob_path)
    try:
        raw = bc.download_blob(max_concurrency=4).readall()
    except Exception:
        return pd.DataFrame()

    txt = raw.decode("utf-8", errors="ignore").strip()
    # JSONL / JSON
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

    # Renommages minimaux selon ton SILVER (job_id, title, status, department/division, locations, created_at, updated_at)
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
            dfm["job_location"] = dfm["locations"].map(_first_location_to_str)
        elif "location" in dfm.columns:
            dfm["job_location"] = dfm["location"]

    # Datetimes usuelles
    for src, dst in [("created_at","job_created_at"), ("updated_at","job_updated_at"),
                     ("published_at","job_published_at"), ("archived_at","job_archived_at"), ("closed_at","job_closed_at")]:
        if dst not in dfm.columns and src in dfm.columns:
            dfm[dst] = dfm[src]
        if dst in dfm.columns:
            dfm[dst] = pd.to_datetime(dfm[dst], errors="coerce", utc=True)

    # Statut → job_open robuste
    status = dfm.get("job_status", "").astype(str).str.lower().str.strip()
    open_terms   = {"open", "active", "published", "live", "visible"}
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
        "job_title", "job_department", "job_location"
    ]
    keep = [c for c in keep if c in dfm.columns]
    dfm = dfm[keep].drop_duplicates(subset=["job_id"] if "job_id" in dfm.columns else None)
    return dfm

# ───────────────────────────── Outcome effectif basé STAGES
STAGE_FIELDS = ["stages", "application_stages", "application_stage_history", "application_stage"]

def _jsonloads_if_json_string(x):
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x

def _extract_stage_strings(v: Any) -> list[str]:
    v = _jsonloads_if_json_string(v)
    out: list[str] = []
    if v is None:
        return out
    if isinstance(v, str):
        out = [p.strip().lower() for p in v.replace(">", ",").split(",") if p.strip()]
    elif isinstance(v, list):
        for item in v:
            if isinstance(item, str):
                out.append(item.strip().lower())
            elif isinstance(item, dict):
                s = item.get("name") or item.get("stage") or item.get("title") or item.get("label") or ""
                if s:
                    out.append(str(s).strip().lower())
            else:
                out.append(str(item).strip().lower())
    elif isinstance(v, dict):
        s = v.get("name") or v.get("stage") or v.get("title") or v.get("label") or ""
        if s:
            out.append(str(s).strip().lower())
    else:
        out.append(str(v).strip().lower())
    return out

def _has_any(labels: Iterable[str], needles: Iterable[str]) -> bool:
    sset = {x.lower() for x in labels}
    for n in needles:
        n = n.lower()
        if any(n in lab for lab in sset):
            return True
    return False

def add_effective_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    stage_col = next((c for c in STAGE_FIELDS if c in df.columns), None)
    if stage_col:
        stage_labels = df[stage_col].map(_extract_stage_strings)
        has_stage_hired = stage_labels.map(lambda labs: _has_any(labs, ["hired"]))
        has_stage_rejected = stage_labels.map(lambda labs: _has_any(labs, ["rejected","declined","not hired"]))
    else:
        has_stage_hired = pd.Series(False, index=df.index)
        has_stage_rejected = pd.Series(False, index=df.index)

    has_hired_date = df["hired_at"].notna() if "hired_at" in df.columns else pd.Series(False, index=df.index)
    has_rejected_date = df["rejected_at"].notna() if "rejected_at" in df.columns else pd.Series(False, index=df.index)

    df = df.copy()
    df["effective_is_rejected"] = (has_stage_rejected | has_rejected_date).astype(bool)
    df["effective_is_hired"] = ((~df["effective_is_rejected"]) & (has_stage_hired | has_hired_date)).astype(bool)

    orig_outcome = df.get("application_outcome", pd.Series("unknown", index=df.index)).fillna("unknown")
    df["application_outcome_effective"] = np.where(
        df["effective_is_rejected"], "rejected",
        np.where(df["effective_is_hired"], "hired", orig_outcome)
    )
    return df

# ───────────────────────────── Helpers UI
def _coalesce(*vals) -> str:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return ""

# ───────────────────────────── UI
st.title("📋 Jobs publiés & candidats — vue cartes")
st.caption("(Applications GOLD) enrichi par (SILVER jobs). Outcome candidat basé sur stages — Rejected > Hired.")

with st.sidebar:
    st.header("⚙️ Source & filtres")
    st.text_input("Container", value=CFG["container"], key="container")
    st.text_input("Blob GOLD (applications)", value=CFG["gold_blob"], key="gold_blob")
    st.text_input("Blob SILVER jobs", value=CFG["jobs_silver_blob"], key="jobs_silver_blob")
    st.divider()
    dev_sample = st.toggle("Charger un échantillon (1000 lignes)", value=False)
    show_only_open = st.toggle("Afficher uniquement les offres ouvertes", value=True)
    sample_rows = 1000 if dev_sample else None

# Chargement
if BlobServiceClient is None:
    st.error("Le paquet `azure-storage-blob` est requis. `pip install azure-storage-blob`")
    st.stop()

try:
    df_apps_raw = load_gold_from_blob(st.session_state["container"], st.session_state["gold_blob"], sample_rows)
except Exception as e:
    st.exception(e); st.stop()

df_jobs_silver = load_jobs_silver_from_blob(st.session_state["container"], st.session_state["jobs_silver_blob"])

if df_apps_raw.empty:
    st.warning("GOLD vide/illisible."); st.stop()

df = add_effective_outcomes(df_apps_raw)

# Filtres haut
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    if "application_created_at" in df.columns and not df["application_created_at"].isna().all():
        min_d = df["application_created_at"].min().date()
        max_d = df["application_created_at"].max().date()
    else:
        min_d, max_d = date(2000, 1, 1), date.today()
    with col1:
        dr = st.date_input("Période candidatures", (min_d, max_d), min_value=min_d, max_value=max_d)
    with col2:
        dept = st.selectbox("Département", ["(Tous)"] + sorted(df.get("job_department", pd.Series(dtype=str)).dropna().unique().tolist()))
    with col3:
        loc = st.selectbox("Localisation", ["(Tous)"] + sorted(df.get("job_location", pd.Series(dtype=str)).dropna().unique().tolist()))
    with col4:
        qtitle = st.text_input("Recherche titre d’offre")

# Appliquer filtres sur les candidatures
mask = pd.Series(True, index=df.index)
if "application_created_at" in df.columns and isinstance(dr, tuple) and len(dr) == 2:
    start_dt = pd.Timestamp(dr[0]).tz_localize("UTC") if pd.Timestamp(dr[0]).tz is None else pd.Timestamp(dr[0])
    end_dt   = (pd.Timestamp(dr[1]).tz_localize("UTC") if pd.Timestamp(dr[1]).tz is None else pd.Timestamp(dr[1])) + pd.Timedelta(days=1)
    mask &= (df["application_created_at"] >= start_dt) & (df["application_created_at"] < end_dt)
if dept != "(Tous)" and "job_department" in df.columns:
    mask &= (df["job_department"] == dept)
if loc != "(Tous)" and "job_location" in df.columns:
    mask &= (df["job_location"] == loc)
if qtitle and "job_title" in df.columns:
    mask &= df["job_title"].fillna("").str.lower().str.contains(qtitle.strip().lower(), na=False)

df_f = df.loc[mask].copy()

# ───────────────────────────── Agrégat par job_id (titres depuis GOLD OU SILVER)
if "job_id" not in df_f.columns:
    st.error("La colonne 'job_id' est absente du GOLD filtré.")
    st.stop()

def _first_nonempty(series: pd.Series):
    for x in series:
        if pd.notna(x) and str(x).strip() != "":
            return x
    return None

jobs_apps = (
    df_f
    .groupby("job_id", dropna=False)
    .agg(
        applications=("application_id", "nunique") if "application_id" in df_f.columns else ("candidate_id", "count"),
        unique_candidates=("candidate_id", "nunique") if "candidate_id" in df_f.columns else ("application_id", "nunique"),
        hires=("effective_is_hired", "sum"),
        rejects=("effective_is_rejected", "sum"),
        first_application=("application_created_at", "min"),
        last_application=("application_created_at", "max"),
        apps_job_title=("job_title", _first_nonempty),
        apps_job_department=("job_department", _first_nonempty),
        apps_job_location=("job_location", _first_nonempty),
    )
    .reset_index()
)

# Merge SILVER (peut apporter title/department/location/status)
if not df_jobs_silver.empty and "job_id" in df_jobs_silver.columns:
    jobs_agg = jobs_apps.merge(df_jobs_silver, on="job_id", how="left")
else:
    jobs_agg = jobs_apps.copy()
    jobs_agg["job_status"] = np.where(
        jobs_agg["applications"] > (jobs_agg["hires"] + jobs_agg["rejects"]),
        "open_heuristic", "closed_heuristic"
    )
    jobs_agg["job_open"] = jobs_agg["job_status"].eq("open_heuristic")

# Coalesce champs d’affichage (GOLD → SILVER)
jobs_agg["job_title_display"] = [
    _coalesce(a, b) for a, b in zip(jobs_agg.get("apps_job_title", ""), jobs_agg.get("job_title", ""))
]
jobs_agg["job_department_display"] = [
    _coalesce(a, b) for a, b in zip(jobs_agg.get("apps_job_department", ""), jobs_agg.get("job_department", ""))
]
jobs_agg["job_location_display"] = [
    _coalesce(a, b) for a, b in zip(jobs_agg.get("apps_job_location", ""), jobs_agg.get("job_location", ""))
]

# Filtrer uniquement les offres ouvertes si demandé
if "job_open" in jobs_agg.columns and show_only_open:
    jobs_agg = jobs_agg[jobs_agg["job_open"].fillna(False)]

# Tri (les plus récentes d’abord)
if "last_application" in jobs_agg.columns:
    jobs_agg = jobs_agg.sort_values("last_application", ascending=False)

# ───────────────────────────── CSS cartes
st.markdown("""
<style>
.card {
  border: 1px solid rgba(128,128,128,0.25);
  border-radius: 14px;
  padding: 12px 14px;
  margin-bottom: 12px;
  background: rgba(125,125,125,0.05);
}
.pills { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 2px 0; }
.pill  { padding:2px 8px; border-radius: 999px; font-size: 0.85rem; border:1px solid rgba(128,128,128,0.25); }
.pill.ok { background: rgba(0,200,83,.15); }
.pill.bad{ background: rgba(255,82,82,.15); }
.pill.mid{ background: rgba(255,193,7,.15); }
.small { opacity: .8; font-size: .9rem; }
.muted { opacity: .7; }
</style>
""", unsafe_allow_html=True)

st.subheader("🧾 Offres (cartes)")

# Grille
n_cols = 3
cols = st.columns(n_cols, gap="large")

def render_candidate_button(r: pd.Series):
    name = r.get("cand_full_name") or f"ID {r.get('candidate_id','?')}"
    stage = r.get("application_stage") or "—"
    out   = r.get("application_outcome_effective") or "unknown"
    dt    = r.get("application_created_at")
    date_s = ("" if pd.isna(dt) else pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M"))
    emoji = "🟢" if out == "hired" else ("🔴" if out == "rejected" else "🕗")
    label = f"{emoji} {name} — {stage} — {date_s}"

    with st.expander(label, expanded=False):
        st.markdown(f"- **Candidate ID**: `{r.get('candidate_id','')}`")
        if r.get("cand_current_title"):
            st.markdown(f"- **Titre actuel**: {r.get('cand_current_title')}")
        st.markdown(f"- **Outcome (effectif)**: `{out}`")
        if r.get("act_had_interview") is not None:
            st.markdown(f"- **Entretien**: `{'oui' if r.get('act_had_interview') else 'non'}`")
        if r.get("y_offer_made") is not None:
            st.markdown(f"- **Offre faite**: `{'oui' if r.get('y_offer_made') else 'non'}`")
        if r.get("rejection_reason"):
            st.markdown(f"- **Rejection reason**: {r.get('rejection_reason')}")

def _fmt_date(ts):
    if isinstance(ts, (pd.Timestamp, datetime)):
        if pd.isna(ts): return ""
        return pd.to_datetime(ts).strftime("%Y-%m-%d")
    return ""

def render_job_card(job_row: pd.Series, idx: int):
    jid   = job_row.get("job_id", "")
    # COALESCE titre/dept/loc (fini les 'Sans titre')
    title = _coalesce(job_row.get("job_title_display"), "Sans titre")
    dept  = _coalesce(job_row.get("job_department_display"), "—")
    loc   = _coalesce(job_row.get("job_location_display"), "—")

    apps  = int(job_row.get("applications", 0))
    hires = int(job_row.get("hires", 0))
    rejects = int(job_row.get("rejects", 0))
    not_placed = max(apps - hires - rejects, 0)

    status = str(job_row.get("job_status", "unknown"))
    job_open = bool(job_row.get("job_open", False))
    badge_class = "ok" if job_open else "bad"

    pub_s   = _fmt_date(job_row.get("job_published_at"))
    arch_s  = _fmt_date(job_row.get("job_archived_at"))
    closed_s= _fmt_date(job_row.get("job_closed_at"))

    # Candidats de ce job (depuis df_f filtré)
    cond = (df_f["job_id"] == jid)
    df_job = df_f.loc[cond].copy().sort_values("application_created_at", ascending=False)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.expander(f"📌 [{jid}] {title}", expanded=False):
            st.markdown(f'<div class="small">Dept: <b>{dept}</b> — Loc: <b>{loc}</b></div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="pills">
                  <div class="pill {badge_class}">Statut: <b>{status}</b></div>
                  <div class="pill">Candidatures: <b>{apps}</b></div>
                  <div class="pill ok">Hired: <b>{hires}</b></div>
                  <div class="pill bad">Rejected: <b>{rejects}</b></div>
                  <div class="pill mid">Toujours pas placé: <b>{not_placed}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if pub_s or arch_s or closed_s:
                st.markdown(
                    f"""<div class="muted">🗓️
                    {"Published: <b>"+pub_s+"</b> • " if pub_s else ""}
                    {"Archived: <b>"+arch_s+"</b> • " if arch_s else ""}
                    {"Closed: <b>"+closed_s+"</b>" if closed_s else ""}
                    </div>""",
                    unsafe_allow_html=True
                )
            st.markdown("---")
            st.markdown("**Candidats**")
            for _, rr in df_job.iterrows():
                render_candidate_button(rr)
        st.markdown('</div>', unsafe_allow_html=True)

# Grille de cartes
if jobs_agg.empty:
    st.info("Aucune offre après filtres.")
else:
    cols = st.columns(n_cols, gap="large")
    for i, (_, r) in enumerate(jobs_agg.iterrows()):
        with cols[i % n_cols]:
            render_job_card(r, i)

# Debug
with st.expander("🛠️ Debug"):
    stage_col = next((c for c in STAGE_FIELDS if c in df.columns), None)
    st.write("Colonne 'stages' détectée :", stage_col or "(aucune)")
    st.write("SILVER jobs chargés :", not df_jobs_silver.empty, "| lignes:", len(df_jobs_silver))
    st.write("Champs coalescés — exemples :")
    st.json(jobs_agg[["job_id","job_title_display","job_department_display","job_location_display"]].head(5).to_dict(orient="records"))
