# app.py — Dashboard GOLD + Scoring (Streamlit)
from __future__ import annotations

import json
import os
from datetime import datetime, date
from typing import Optional, Iterable, Any

import numpy as np
import pandas as pd
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go

# Azure Blob
try:
    from azure.storage.blob import BlobServiceClient  # type: ignore
except Exception:
    BlobServiceClient = None

st.set_page_config(page_title="CV Compatibility — Dashboard GOLD", page_icon="✅", layout="wide")

# Configuration API
API_URL = os.getenv("API_URL", "http://localhost:8000")

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
    col5, col6 = st.columns(2)
    with col5:
        title_query = st.text_input("Recherche titre d'offre (contient)")
    with col6:
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
    "job_id", "job_title", "job_department",
    "candidate_id", "cand_full_name", "cand_current_title",
    "effective_is_hired", "effective_is_rejected",
    "y_offer_made", "act_had_interview", "rejection_reason",
]
cols_show = [c for c in priority_cols if c in df_f.columns] + [c for c in df_f.columns if c not in priority_cols]
st.dataframe(df_f[cols_show].head(1000), use_container_width=True, hide_index=True)

# --- Onglets ---
tab1, tab2, tab3 = st.tabs(["📊 Vue d'ensemble & Export", "📈 Graphiques & Analyse", "🤖 Scoring Rapide"])

with tab1:
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

with tab2:
    st.subheader("📈 Analyse Graphique des Données")

    # Palette de couleurs violette du thème
    violet_palette = ["#DDD3FD", "#BCA7FA", "#9A7BF8", "#5525E4", "#4321A6", "#2D1574", "#1E1047"]
    primary_violet = "#5525E4"
    dark_violet = "#2D1574"

    if not df_f.empty:
        # Graphique 1: Evolution temporelle des candidatures
        if "application_created_at" in df_f.columns:
            st.markdown("#### 📅 Candidatures par jour")
            df_by_date = df_f.groupby(df_f["application_created_at"].dt.date).size().reset_index(name="count")
            fig_timeline = px.line(
                df_by_date,
                x="application_created_at",
                y="count",
                title="Nombre de candidatures par jour",
                labels={"application_created_at": "Date", "count": "Nombre de candidatures"},
                markers=True
            )
            fig_timeline.update_traces(line=dict(color=primary_violet, width=3), marker=dict(size=8, color=primary_violet))
            fig_timeline.update_layout(height=400, hovermode="x unified")
            st.plotly_chart(fig_timeline, use_container_width=True)

        # Graphique 2: Top offres (job titles)
        if "job_title" in df_f.columns:
            st.markdown("#### 💼 Top 10 Offres")
            top_jobs = df_f["job_title"].value_counts().head(10).reset_index()
            top_jobs.columns = ["Job Title", "Count"]

            fig_jobs = px.bar(
                top_jobs,
                x="Count",
                y="Job Title",
                orientation="h",
                title="Offres avec le plus de candidatures",
                color="Count",
                color_continuous_scale=violet_palette
            )
            fig_jobs.update_layout(height=400, coloraxis_colorbar=dict(title="Nombre"))
            fig_jobs.update_traces(marker=dict(line=dict(color=dark_violet, width=0.5)))
            st.plotly_chart(fig_jobs, use_container_width=True)

        # Graphique 3: Taux de succès (Hired) par département
        if "job_department" in df_f.columns and "effective_is_hired" in df_f.columns:
            st.markdown("#### 🎯 Taux de succès par département")

            # Calculer le taux de hired par département
            dept_stats = df_f.groupby("job_department").agg({
                "effective_is_hired": ["sum", "count"]
            }).reset_index()
            dept_stats.columns = ["Department", "Hired", "Total"]
            dept_stats["Taux de succès (%)"] = (dept_stats["Hired"] / dept_stats["Total"] * 100).round(1)

            # Filtrer les départements avec au moins 5 candidatures pour avoir des stats significatives
            dept_stats = dept_stats[dept_stats["Total"] >= 5].sort_values("Taux de succès (%)", ascending=False).head(10)

            if not dept_stats.empty:
                fig_success_rate = px.bar(
                    dept_stats,
                    x="Department",
                    y="Taux de succès (%)",
                    title="Taux de recrutement par département (min. 5 candidatures)",
                    color="Taux de succès (%)",
                    color_continuous_scale=violet_palette,
                    hover_data={"Total": True, "Hired": True}
                )
                fig_success_rate.update_layout(height=400, xaxis_tickangle=-45, coloraxis_colorbar=dict(title="Taux (%)"))
                fig_success_rate.update_traces(marker=dict(line=dict(color=dark_violet, width=0.5)))
                st.plotly_chart(fig_success_rate, use_container_width=True)

        # Graphique 4: Distribution Outcome (Pie chart simplifié)
        if "application_outcome_effective" in df_f.columns:
            st.markdown("#### 📊 Distribution des résultats")
            outcome_counts = df_f["application_outcome_effective"].value_counts().reset_index()
            outcome_counts.columns = ["Outcome", "Count"]

            fig_outcome = px.pie(
                outcome_counts,
                names="Outcome",
                values="Count",
                title="Répartition Hired / Rejected / Autres",
                hole=0.4,
                color_discrete_sequence=violet_palette[:len(outcome_counts)]
            )
            fig_outcome.update_traces(textposition="inside", textinfo="label+percent+value")
            st.plotly_chart(fig_outcome, use_container_width=True)
    else:
        st.warning("❌ Aucune donnée disponible pour cette sélection")

with tab3:
    st.subheader("🤖 Scoring Rapide d'une Paire Job/CV")
    st.info("💡 Testez l'API de scoring pour calculer la compatibilité entre un job et un CV")

    # Vérifier que l'API est accessible
    try:
        api_health = requests.get(f"{API_URL}/healthz", timeout=5)
        api_available = api_health.status_code == 200
    except:
        api_available = False

    if not api_available:
        st.error(f"❌ API non accessible à {API_URL}. Assurez-vous que le serveur FastAPI est en cours d'exécution.")
        st.code("python api/main.py", language="bash")
    else:
        st.success("✅ API accessible")

        # Deux modes de scoring
        scoring_mode = st.radio("Mode de scoring", ["📝 Formulaire texte", "🔗 Depuis les données GOLD"])

        if scoring_mode == "📝 Formulaire texte":
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 💼 Informations du Job")
                job_title = st.text_input("Titre du poste", "Senior Data Engineer")
                job_desc = st.text_area(
                    "Description du job",
                    "We are looking for a senior data engineer with 5+ years of experience. Required: Python, Spark, Airflow, AWS.",
                    height=150
                )
                job_skills = st.text_input("Compétences requises (séparées par virgule)", "Python, Spark, Airflow, AWS")
                job_years = st.number_input("Années d'expérience requises", min_value=0, value=5)
                job_languages = st.multiselect("Langues requises", ["Français", "Anglais", "Autre"], default=["Français"])

            with col2:
                st.markdown("### 👤 Informations du CV")
                cv_text = st.text_area(
                    "Résumé du CV / Expérience",
                    "Senior Data Engineer with 7 years of experience. Expertise in Python, Spark, Airflow, AWS, Kubernetes. Worked on ETL pipelines and data warehousing.",
                    height=150
                )
                cv_skills = st.text_input("Compétences du candidat (séparées par virgule)", "Python, Spark, Airflow, AWS, Kubernetes, Docker")
                cv_years = st.number_input("Années d'expérience du candidat", min_value=0, value=7)
                cv_languages = st.multiselect("Langues du candidat", ["Français", "Anglais", "Autre"], default=["Anglais"])

            if st.button("🚀 Calculer le score", use_container_width=True):
                with st.spinner("⏳ Calcul du score en cours..."):
                    try:
                        payload = {
                            "job": {
                                "job_title": job_title,
                                "job_description_text": job_desc,
                                "job_required_skills_plus": [s.strip() for s in job_skills.split(",")],
                                "job_required_years_num": int(job_years),
                                "job_languages_required": job_languages,
                            },
                            "cv": {
                                "text_cv_full": cv_text,
                                "text_cv_skills": cv_skills,
                                "cv_years_experience_num": int(cv_years),
                                "cv_languages": cv_languages,
                            }
                        }

                        response = requests.post(
                            f"{API_URL}/score",
                            json=payload,
                            params={"include_features": True},
                            timeout=120
                        )

                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Score calculé avec succès!")

                            # Afficher les scores
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Score Global", f"{result.get('score_global', 0):.3f}")
                            with col2:
                                st.metric("Score Type", f"{result.get('score_type', 'N/A')}" if result.get('score_type') else "N/A")
                            with col3:
                                st.metric("Score Final", f"{result.get('score_final', 0):.3f}")
                            with col4:
                                pred_text = "✅ ACCEPTÉ" if result.get('pred') == 1 else "❌ REJETÉ"
                                st.metric("Prédiction", pred_text)

                            # Afficher les features
                            st.markdown("### 📊 Features Calculées (21 features)")
                            features = result.get("features_used", {})

                            # Regrouper par catégorie
                            categories = {
                                "Similarité (4)": ["sim_dense_full", "sim_sparse_tfidf", "sim_exp_title", "sim_exp_responsibilities"],
                                "Skills (4)": ["skill_gap_ratio", "skill_gap_per_year_required", "cov_soft_skills", "critical_skills_blocker"],
                                "Expérience (3)": ["job_required_years_num", "exp_gap_years", "overqualification_penalty"],
                                "Seniority (1)": ["seniority_gap"],
                                "Langues (1)": ["lang_required_coverage"],
                                "École (1)": ["ecole"],
                                "Composites (2)": ["sim_x_seniority", "soft_skills_weighted"],
                                "Positionnement (1)": ["rank_in_offer_pool_norm"],
                                "Sélectivité (2)": ["job_selectivity_historical", "job_competition_index_norm"],
                                "Phase 4J (2)": ["is_low_selectivity_boost", "job_candidate_mismatch_flags"],
                            }

                            for category, feat_names in categories.items():
                                with st.expander(f"{category}"):
                                    feature_cols = st.columns(2)
                                    for i, feat_name in enumerate(feat_names):
                                        col = feature_cols[i % 2]
                                        if feat_name in features:
                                            col.metric(feat_name, f"{features[feat_name]:.4f}")
                                        else:
                                            col.warning(f"❌ {feat_name} manquante")
                        else:
                            st.error(f"❌ Erreur API: {response.status_code}")
                            st.code(response.text)
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Timeout: La requête a pris trop de temps (>120s)")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")

        else:  # Mode GOLD
            st.markdown("### Sélectionner une paire depuis GOLD")

            if not df_f.empty:
                # Sélectionner une application
                selected_app_id = st.selectbox(
                    "Sélectionner une candidature",
                    df_f["application_id"].unique() if "application_id" in df_f.columns else [],
                    format_func=lambda x: f"App {x} - {df_f[df_f['application_id'] == x]['job_title'].values[0] if 'job_title' in df_f.columns else 'N/A'}"
                )

                if selected_app_id:
                    app_row = df_f[df_f["application_id"] == selected_app_id].iloc[0]

                    st.info(f"📋 Application: {selected_app_id}")

                    if st.button("🚀 Scorer cette paire", use_container_width=True):
                        with st.spinner("⏳ Calcul du score..."):
                            try:
                                # Construire le payload à partir des données GOLD
                                payload = {
                                    "pair_id": selected_app_id,
                                    "job": {
                                        "job_id": app_row.get("job_id", ""),
                                        "job_title": app_row.get("job_title", ""),
                                        "job_description_text": app_row.get("job_description_text", ""),
                                    },
                                    "cv": {
                                        "candidate_id": app_row.get("candidate_id", ""),
                                        "text_cv_full": app_row.get("text_cv_full", ""),
                                    }
                                }

                                response = requests.post(
                                    f"{API_URL}/score",
                                    json=payload,
                                    params={"include_features": True},
                                    timeout=120
                                )

                                if response.status_code == 200:
                                    result = response.json()
                                    st.success("✅ Score calculé!")

                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Score Global", f"{result.get('score_global', 0):.3f}")
                                    with col2:
                                        st.metric("Score Type", f"{result.get('score_type', 'N/A')}" if result.get('score_type') else "N/A")
                                    with col3:
                                        st.metric("Score Final", f"{result.get('score_final', 0):.3f}")
                                    with col4:
                                        pred_text = "✅ ACCEPTÉ" if result.get('pred') == 1 else "❌ REJETÉ"
                                        st.metric("Prédiction", pred_text)
                                else:
                                    st.error(f"❌ Erreur API: {response.status_code}")
                            except Exception as e:
                                st.error(f"❌ Erreur: {str(e)}")
            else:
                st.warning("❌ Aucune candidature disponible avec les filtres sélectionnés")
