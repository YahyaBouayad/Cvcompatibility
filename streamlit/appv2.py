# app.py — Dashboard GOLD (Streamlit)
from __future__ import annotations

import json
import os
from datetime import datetime, date
from typing import Optional, Iterable, Any

import numpy as np
import pandas as pd
import streamlit as st

# Azure Blob
try:
    from azure.storage.blob import BlobServiceClient  # type: ignore
except Exception:
    BlobServiceClient = None

st.set_page_config(page_title="CV Compatibility — Dashboard GOLD", page_icon="✅", layout="wide")

ENV_AZ = {
    "connection_string": "DefaultEndpointsProtocol=https;AccountName=absaifabrik;AccountKey=dHnt6TqjK6GYHvEjLTKenFdRHitSCByxcqenpCAvP/+GkY6XjHk7+BMfSpVuhbSpUi/5EfGq61CR+AStw0NiCA==;EndpointSuffix=core.windows.net",
    "account_url": os.getenv("AZURE_BLOB_ACCOUNT_URL"),
    "sas_token": os.getenv("AZURE_BLOB_SAS_TOKEN"),
    "account_key": os.getenv("AZURE_STORAGE_ACCOUNT_KEY"),
    "container": os.getenv("AZURE_BLOB_CONTAINER") or "cvcompat",
    "gold_blob": os.getenv("GOLD_BLOB_PATH") or "gold/applications_gold_latest.jsonl",
}

# secrets > env > défauts
CFG = {
    "connection_string":  ENV_AZ["connection_string"],
    "account_url": ENV_AZ["account_url"],
    "sas_token":  ENV_AZ["sas_token"],
    "account_key":  ENV_AZ["account_key"],
    "container":  ENV_AZ["container"],
    "gold_blob":  ENV_AZ["gold_blob"],
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

    if "application_outcome" in df.columns:
        df["application_outcome"] = df["application_outcome"].fillna("unknown")

    # Ne PAS se baser sur is_hired ; on ne touche pas ici.
    for b in ("is_rejected", "y_hired", "y_offer_made", "act_had_interview"):
        if b in df.columns:
            df[b] = df[b].fillna(False).astype(bool)

    return df

def _percent(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0

def _serialize_value(v: Any):
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return None if pd.isna(v) else v.isoformat()
    try:
        if pd.isna(v): return None
    except Exception:
        pass
    return v

def _download_bytes_from_df(df: pd.DataFrame, kind: str = "csv") -> bytes:
    if kind == "csv":
        return df.to_csv(index=False).encode("utf-8")
    if kind == "jsonl":
        recs = []
        for _, row in df.iterrows():
            recs.append({k: _serialize_value(v) for k, v in row.items()})
        jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in recs)
        return jsonl.encode("utf-8")
    raise ValueError("kind doit être 'csv' ou 'jsonl'.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Paramètres & Source")
    st.caption("Azure Blob → GOLD JSONL")
    st.text_input("Container", value=CFG["container"], key="container")
    st.text_input("Blob path (GOLD)", value=CFG["gold_blob"], key="blob_path")
    st.divider()
    dev_sample = st.toggle("Charger un échantillon (1000 lignes)", value=False)
    sample_rows = 1000 if dev_sample else None
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
            },
            language="json",
        )

# --- Load data ---
st.title("🏠 Dashboard GOLD — Vue d’ensemble")
st.caption("Source: Azure Blob → GOLD JSONL")

if BlobServiceClient is None:
    st.error("Le paquet `azure-storage-blob` est requis. `pip install azure-storage-blob`")
    st.stop()

try:
    df = load_gold_from_blob(st.session_state["container"], st.session_state["blob_path"], sample_rows)
except Exception as e:
    st.exception(e); st.stop()

if df.empty:
    st.warning("Le GOLD est vide ou illisible."); st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  RÈGLE MÉTIER DEMANDÉE : outcome basé sur 'stages' (Rejected > Hired)
# ─────────────────────────────────────────────────────────────────────────────
STAGE_FIELDS = ["stages", "application_stages", "application_stage_history", "application_stage"]
stage_col = next((c for c in STAGE_FIELDS if c in df.columns), None)

def _jsonloads_if_json_string(x):
    # Si 'stages' est stocké sous forme de string JSON, on le parse.
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x

def _extract_stage_strings(v: Any) -> list[str]:
    """Retourne une liste de labels de stages en minuscules."""
    v = _jsonloads_if_json_string(v)
    out: list[str] = []
    if v is None:
        return out
    if isinstance(v, str):
        # Exemple: "Applied > Interview > Hired"
        out = [p.strip().lower() for p in v.replace(">", ",").split(",") if p.strip()]
    elif isinstance(v, list):
        for item in v:
            if isinstance(item, str):
                out.append(item.strip().lower())
            elif isinstance(item, dict):
                # supporte clés fréquentes
                s = item.get("name") or item.get("stage") or item.get("title") or item.get("label") or ""
                if s:
                    out.append(str(s).strip().lower())
            else:
                out.append(str(item).strip().lower())
    elif isinstance(v, dict):
        # un dict unique
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
        # match exact ou substring (au cas où "hired (moved)")
        if any(n in lab for lab in sset):
            return True
    return False

if stage_col:
    stage_labels = df[stage_col].map(_extract_stage_strings)
    has_stage_hired = stage_labels.map(lambda labs: _has_any(labs, ["hired"]))
    has_stage_rejected = stage_labels.map(lambda labs: _has_any(labs, ["rejected", "declined", "not hired"]))
else:
    # Fallback si pas de colonne stages — on ne se base PAS sur is_hired, seulement sur dates
    has_stage_hired = pd.Series(False, index=df.index)
    has_stage_rejected = pd.Series(False, index=df.index)

# Fallback dates si pas de 'stages' renseigné sur la ligne
has_hired_date = df["hired_at"].notna() if "hired_at" in df.columns else pd.Series(False, index=df.index)
has_rejected_date = df["rejected_at"].notna() if "rejected_at" in df.columns else pd.Series(False, index=df.index)

# Priorité: Rejected > Hired
df["effective_is_rejected"] = (has_stage_rejected | has_rejected_date).astype(bool)
df["effective_is_hired"] = ((~df["effective_is_rejected"]) & (has_stage_hired | has_hired_date)).astype(bool)

# Outcome effectif final
orig_outcome = df.get("application_outcome", pd.Series("unknown", index=df.index)).fillna("unknown")
df["application_outcome_effective"] = np.where(
    df["effective_is_rejected"], "rejected",
    np.where(df["effective_is_hired"], "hired", orig_outcome)
)

# --- Filtres ---
with st.container():
    st.subheader("🔎 Filtres")
    col1, col2, col3, col4 = st.columns(4)

    if "application_created_at" in df.columns and not df["application_created_at"].isna().all():
        min_d = df["application_created_at"].min().date()
        max_d = df["application_created_at"].max().date()
    else:
        min_d, max_d = date(2000, 1, 1), date.today()

    with col1:
        dr = st.date_input("Période (création)", (min_d, max_d), min_value=min_d, max_value=max_d)

    stages_list = sorted([x for x in df.get("application_stage", pd.Series(dtype=str)).dropna().unique().tolist() if x != ""])
    outcomes = sorted([x for x in df.get("application_outcome_effective", pd.Series(dtype=str)).dropna().unique().tolist() if x != ""])
    with col2:
        selected_stages = st.multiselect("Stages (courant)", stages_list)
    with col3:
        selected_outcomes = st.multiselect("Outcome (effectif)", outcomes)

    with col4:
        dept = st.selectbox("Département", ["(Tous)"] + sorted(df.get("job_department", pd.Series(dtype=str)).dropna().unique().tolist()))
    col5, col6, col7 = st.columns(3)
    with col5:
        loc = st.selectbox("Localisation", ["(Tous)"] + sorted(df.get("job_location", pd.Series(dtype=str)).dropna().unique().tolist()))
    with col6:
        title_query = st.text_input("Recherche titre d’offre (contient)")
    with col7:
        only_has_interview = st.toggle("Uniquement avec entretien", value=False)

mask = pd.Series(True, index=df.index)

if "application_created_at" in df.columns and isinstance(dr, tuple) and len(dr) == 2:
    start_dt = pd.Timestamp(dr[0]).tz_localize("UTC") if pd.Timestamp(dr[0]).tz is None else pd.Timestamp(dr[0])
    end_dt = (pd.Timestamp(dr[1]).tz_localize("UTC") if pd.Timestamp(dr[1]).tz is None else pd.Timestamp(dr[1])) + pd.Timedelta(days=1)
    mask &= (df["application_created_at"] >= start_dt) & (df["application_created_at"] < end_dt)

if selected_stages and "application_stage" in df.columns:
    mask &= df["application_stage"].isin(selected_stages)

if selected_outcomes and "application_outcome_effective" in df.columns:
    mask &= df["application_outcome_effective"].isin(selected_outcomes)

if dept != "(Tous)" and "job_department" in df.columns:
    mask &= (df["job_department"] == dept)

if loc != "(Tous)" and "job_location" in df.columns:
    mask &= (df["job_location"] == loc)

if title_query and "job_title" in df.columns:
    q = title_query.strip().lower()
    mask &= df["job_title"].fillna("").str.lower().str.contains(q, na=False)

if only_has_interview and "act_had_interview" in df.columns:
    mask &= df["act_had_interview"] == True

df_f = df.loc[mask].copy()

# --- KPIs (effectifs) ---
st.subheader("📈 Indicateurs clés")
total_apps = len(df_f)
n_hired = int(df_f.get("effective_is_hired", pd.Series(False, index=df_f.index)).sum())
n_rejected = int(df_f.get("effective_is_rejected", pd.Series(False, index=df_f.index)).sum())
n_offer = int(df_f.get("y_offer_made", pd.Series(False, index=df_f.index)).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidatures", f"{total_apps:,}")
c2.metric("Hirings (effectif)", f"{n_hired:,}", f"{_percent(n_hired, total_apps):.1f}%")
c3.metric("Rejets (effectif)", f"{n_rejected:,}", f"{_percent(n_rejected, total_apps):.1f}%")
c4.metric("Offres faites", f"{n_offer:,}", f"{_percent(n_offer, total_apps):.1f}%")

# --- Table ---
st.subheader("📄 Données filtrées")
priority_cols = [
    "application_id", "application_created_at", "application_stage",
    "application_outcome_effective", "application_outcome",
    "job_id", "job_title", "job_department", "job_location",
    "candidate_id", "cand_full_name", "cand_current_title",
    "effective_is_hired", "effective_is_rejected",
    "y_offer_made", "act_had_interview", "rejection_reason",
]
cols_show = [c for c in priority_cols if c in df_f.columns] + [c for c in df_f.columns if c not in priority_cols]
st.dataframe(df_f[cols_show].head(1000), use_container_width=True, hide_index=True)

# --- Exports ---
st.subheader("⬇️ Export")
colx, coly, colz = st.columns(3)
with colx:
    st.download_button(
        "Télécharger (CSV)",
        data=_download_bytes_from_df(df_f, "csv"),
        file_name="applications_gold_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
with coly:
    st.download_button(
        "Télécharger (JSONL)",
        data=_download_bytes_from_df(df_f, "jsonl"),
        file_name="applications_gold_filtered.jsonl",
        mime="application/x-ndjson",
        use_container_width=True,
    )
with colz:
    st.caption(f"Colonnes: {len(df_f.columns)} | Lignes filtrées: {len(df_f):,}")

# --- Debug ---
with st.expander("🛠️ Aide & Debug"):
    st.write("**Colonne utilisée pour stages:**", stage_col or "(aucune trouvée)")
    if stage_col:
        st.write("Exemples de stages (premières lignes):")
        st.write(df[stage_col].head(5))
    st.write("**Colonnes disponibles (extrait):**")
    st.code(df.columns.tolist(), language="python")
    st.write("**dtypes:**"); st.write(df_f.dtypes.astype(str))
    if "application_created_at" in df_f.columns and not df_f.empty:
        st.write("**Dates min/max (application_created_at):**", df_f["application_created_at"].min(), "→", df_f["application_created_at"].max())
    st.write("**Exemple de ligne (préparée pour JSON):**")
    st.json({k: _serialize_value(v) for k, v in (df_f.iloc[0].to_dict() if not df_f.empty else {}).items()})

st.success("✅ Règle outcome=stages appliquée (Rejected > Hired) + export JSONL compatible.", icon="✅")
