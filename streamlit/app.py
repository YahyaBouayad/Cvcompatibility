import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from azure.storage.blob import BlobServiceClient

# =========================
#   CONFIG STREAMLIT
# =========================
st.set_page_config(page_title="CV Compatibility — Jobs publics", page_icon="🧩", layout="wide")
st.title("🧩 CV Compatibility — Jobs publics (POC Streamlit)")
st.caption("Lecture directe des fichiers *silver* (JSONL) depuis Azure Blob Storage, sans appel API Teamtailor.")

# =========================
#   HELPERS BLOB / JSONL
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def get_blob_bytes(connection_string: str, container: str, blob_path: str) -> bytes:
    bsc = BlobServiceClient.from_connection_string(connection_string)
    return bsc.get_container_client(container).get_blob_client(blob_path).download_blob().readall()

def _coerce_json(obj: Any) -> List[Dict[str, Any]]:
    """
    Supporte:
      - liste[dict]
      - dict (-> liste)
      - JSON (texte)
      - JSONL (une ligne = un objet)
      - bytes
    Retourne une liste de dicts.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    text = obj.decode("utf-8", errors="ignore") if isinstance(obj, (bytes, bytearray)) else (obj or "")
    if not isinstance(text, str):
        return []

    # Essai JSON complet
    try:
        parsed = json.loads(text)
        return _coerce_json(parsed)
    except json.JSONDecodeError:
        pass

    # Fallback JSONL
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # ignore
            pass
    return rows

@st.cache_data(show_spinner=True, ttl=300)
def load_json_from_blob(connection_string: str, container: str, blob_path: str) -> List[Dict[str, Any]]:
    raw = get_blob_bytes(connection_string, container, blob_path)
    return _coerce_json(raw)

def pick(row: pd.Series, key: str, default=None):
    return row[key] if key in row and pd.notna(row[key]) else default

def to_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val)

def join_nonempty(values: List[Any], sep=" ") -> str:
    return sep.join([to_str(v).strip() for v in values if to_str(v).strip()])

def csvify_list(val) -> str:
    if isinstance(val, list):
        return ", ".join(map(to_str, val))
    return to_str(val)

# =========================
#   NORMALISATION (ADAPTÉE À TA STRUCTURE)
# =========================
def normalize_jobs(jobs_raw: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(jobs_raw, sep=".")
    # champs exacts présents dans ton JSONL
    df["job_id"]   = df.apply(lambda r: pick(r, "job_id"), axis=1)
    df["title"]    = df.apply(lambda r: pick(r, "title"), axis=1)
    df["status"]   = df.apply(lambda r: pick(r, "status", ""), axis=1)

    # description = body_text
    df["description"] = df.apply(lambda r: pick(r, "body_text", ""), axis=1)

    # infos utiles d’affichage (optionnelles)
    df["locations"] = df.apply(lambda r: csvify_list(pick(r, "locations", [])), axis=1)
    df["tags"]      = df.apply(lambda r: csvify_list(pick(r, "tags", [])), axis=1)
    df["department"]= df.apply(lambda r: pick(r, "department", ""), axis=1)
    df["division"]  = df.apply(lambda r: pick(r, "division", ""), axis=1)
    df["created_at"]= df.apply(lambda r: pick(r, "created_at", ""), axis=1)
    df["updated_at"]= df.apply(lambda r: pick(r, "updated_at", ""), axis=1)

    keep = ["job_id","title","status","description","locations","tags","department","division","created_at","updated_at"]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]

def normalize_apps(apps_raw: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(apps_raw, sep=".")
    # clés exactement présentes dans ton JSONL job-application
    df["application_id"] = df.apply(lambda r: pick(r, "application_id"), axis=1)
    df["job_id"]         = df.apply(lambda r: pick(r, "job_id"), axis=1)
    df["candidate_id"]   = df.apply(lambda r: pick(r, "candidate_id"), axis=1)
    df["created_at"]     = df.apply(lambda r: pick(r, "created_at"), axis=1)
    df["updated_at"]     = df.apply(lambda r: pick(r, "updated_at"), axis=1)
    df["stage_name"]     = df.apply(lambda r: pick(r, "stage_name"), axis=1)
    df["status"]         = df.apply(lambda r: pick(r, "status"), axis=1)
    df["decision"]       = df.apply(lambda r: pick(r, "decision"), axis=1)
    df["rejected_at"]    = df.apply(lambda r: pick(r, "rejected_at"), axis=1)
    df["reject_reason"]  = df.apply(lambda r: pick(r, "reject_reason_text"), axis=1)
    df["source_site"]    = df.apply(lambda r: pick(r, "source_site"), axis=1)
    df["source_url"]     = df.apply(lambda r: pick(r, "source_url"), axis=1)
    df["sourced"]        = df.apply(lambda r: pick(r, "sourced"), axis=1)
    df["changed_stage_at"]= df.apply(lambda r: pick(r, "changed_stage_at"), axis=1)
    df["cover_letter_present"]= df.apply(lambda r: pick(r, "cover_letter_present"), axis=1)

    return df[[
        "application_id","job_id","candidate_id","created_at","updated_at",
        "stage_name","status","decision","rejected_at","reject_reason",
        "source_site","source_url","sourced","changed_stage_at","cover_letter_present"
    ]]

def normalize_candidates(cands_raw: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(cands_raw, sep=".")

    df["candidate_id"] = df.apply(lambda r: pick(r, "candidate_id"), axis=1)
    df["first_name"]   = df.apply(lambda r: pick(r, "first_name", ""), axis=1)
    df["last_name"]    = df.apply(lambda r: pick(r, "last_name", ""), axis=1)
    df["full_name"]    = [join_nonempty([f, l]) for f, l in zip(df["first_name"], df["last_name"])]

    df["email"]        = df.apply(lambda r: pick(r, "email"), axis=1)
    df["phone"]        = df.apply(lambda r: pick(r, "phone"), axis=1)
    df["location"]     = df.apply(lambda r: pick(r, "location"), axis=1)
    df["cv.profile.location"]  = df.apply(lambda r: pick(r, "cv.profile.location"), axis=1)
    df["linkedin_url"] = df.apply(lambda r: pick(r, "linkedin_url"), axis=1)
    df["picture_url"]  = df.apply(lambda r: pick(r, "picture_url"), axis=1)
    df["created_at"]   = df.apply(lambda r: pick(r, "created_at"), axis=1)
    df["updated_at"]   = df.apply(lambda r: pick(r, "updated_at"), axis=1)
    df["source_site"]  = df.apply(lambda r: pick(r, "source_site"), axis=1)
    df["referring_url"]= df.apply(lambda r: pick(r, "referring_url"), axis=1)
    df["internal"]     = df.apply(lambda r: pick(r, "internal"), axis=1)
    df["sourced"]      = df.apply(lambda r: pick(r, "sourced"), axis=1)
    df["unsubscribed"] = df.apply(lambda r: pick(r, "unsubscribed"), axis=1)

    keep = ["candidate_id","full_name","location","cv.profile.location","email","phone","linkedin_url","picture_url","created_at","updated_at","source_site","referring_url","internal","sourced","unsubscribed"]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]

def is_public_row(row: pd.Series) -> bool:
    # Avec ton JSONL: on considère public si status ∈ {published, public, open, active}
    status = str(row.get("status", "") or "").lower()
    return any(s in status for s in ["public", "published", "open", "active"])

def shorten_md(md: str, max_chars: int = 500) -> Tuple[str, bool]:
    if md is None:
        return "", False
    text = str(md)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip() + "…", True

# =========================
#   SIDEBAR CONFIG
# =========================
st.sidebar.header("⚙️ Configuration Blob (lecture)")
conn_default = "DefaultEndpointsProtocol=https;AccountName=absaifabrik;AccountKey=dHnt6TqjK6GYHvEjLTKenFdRHitSCByxcqenpCAvP/+GkY6XjHk7+BMfSpVuhbSpUi/5EfGq61CR+AStw0NiCA==;EndpointSuffix=core.windows.net"
container_default = "cvcompat"

connection_string = st.sidebar.text_input("Connection string Azure", conn_default, type="password")
container_name = st.sidebar.text_input("Nom du container", container_default)

st.sidebar.subheader("📄 Fichiers Silver (JSONL)")
blob_jobs = st.sidebar.text_input("Jobs", "silver/jobs/2025-08-26T09-37-29Z.jsonl")
blob_apps = st.sidebar.text_input("Job applications", "silver/job-applications/2025-08-26T12-41-55Z.jsonl")
blob_candidates = st.sidebar.text_input("Candidates unifié", "silver/candidates_unified/2025-08-22T13-56-40Z.jsonl")

with st.sidebar.expander("🔎 Filtres / Affichage", expanded=True):
    search_query = st.text_input("Recherche dans titre/description", "")
    sort_by = st.selectbox("Trier par", ["Titre (A→Z)", "Nb candidatures (desc)", "Nb candidatures (asc)"])
    per_row = st.slider("Cartes par ligne", min_value=1, max_value=4, value=2)
    show_pii = st.checkbox("Afficher e-mail / téléphone (PII)", value=False)

# =========================
#   LOAD & PREP DATA
# =========================
if not connection_string or not container_name:
    st.info("➡️ Renseigne la **connection string** et le **container** dans la sidebar pour charger les données.")
    st.stop()

try:
    jobs_raw = load_json_from_blob(connection_string, container_name, blob_jobs)
    apps_raw = load_json_from_blob(connection_string, container_name, blob_apps)
    cands_raw = load_json_from_blob(connection_string, container_name, blob_candidates)
except Exception as e:
    st.error(f"Échec de lecture des blobs. Détails: {e}")
    st.stop()

df_jobs = normalize_jobs(jobs_raw)
df_apps = normalize_apps(apps_raw)
df_cands = normalize_candidates(cands_raw)


# Filtre public
df_jobs_public = df_jobs[df_jobs.apply(is_public_row, axis=1)].copy()

# Agrégation candidatures
apps_count = (
    df_apps.dropna(subset=["job_id"])
          .groupby("job_id", dropna=True)
          .size()
          .rename("applicants_count")
          .reset_index()
)

# Join pour vue par job
df_jobs_view = df_jobs_public.merge(apps_count, on="job_id", how="left")
df_jobs_view["applicants_count"] = df_jobs_view["applicants_count"].fillna(0).astype(int)

# Recherche globale job
if search_query.strip():
    q = search_query.strip().lower()
    mask = (
        df_jobs_view["title"].astype(str).str.lower().str.contains(q, na=False)
        | df_jobs_view["description"].astype(str).str.lower().str.contains(q, na=False)
    )
    df_jobs_view = df_jobs_view[mask]

# Tri
if sort_by == "Titre (A→Z)":
    df_jobs_view = df_jobs_view.sort_values(by=["title", "job_id"], na_position="last")
elif sort_by == "Nb candidatures (desc)":
    df_jobs_view = df_jobs_view.sort_values(by=["applicants_count", "title"], ascending=[False, True])
elif sort_by == "Nb candidatures (asc)":
    df_jobs_view = df_jobs_view.sort_values(by=["applicants_count", "title"], ascending=[True, True])

st.success(f"✅ {len(df_jobs_view)} job(s) public(s) chargé(s).")

# =========================
#   CANDIDATES TABLE BUILDER
# =========================
def build_candidates_table_for_job(job_id: Any, local_filter: str = "") -> pd.DataFrame:
    apps = df_apps[df_apps["job_id"].astype(str) == str(job_id)].copy()
    if apps.empty:
        return pd.DataFrame(columns=[
            "candidate_id","Nom","Localisation","Source","Date candidature","Stage","Statut","Décision",
            "Rejeté le","Raison rejet","LinkedIn","Email","Téléphone"
        ])

    merged = apps.merge(df_cands, on="candidate_id", how="left", suffixes=("", "_cand"))

    table = pd.DataFrame({
        "candidate_id": merged["candidate_id"],
        "Nom": merged["full_name"],
        "Localisation": merged["location"],
        "CV_location": merged["cv.profile.location"],
        "Source": merged["source_site"],
        "Date candidature": merged["created_at"],      # côté application
        "Stage": merged["stage_name"],
        "Statut": merged["status"],
        "Décision": merged["decision"],
        "Rejeté le": merged["rejected_at"],
        "Raison rejet": merged["reject_reason"],
        "LinkedIn": merged["linkedin_url"],
        "Email": merged["email"],
        "Téléphone": merged["phone"],
    })

    # Filtre local
    if local_filter.strip():
        q = local_filter.lower()
        filt_cols = ["Nom","Localisation","CV_location","Source","Stage","Statut","Décision","Raison rejet"]
        mask = pd.Series([False]*len(table))
        for c in filt_cols:
            mask = mask | table[c].astype(str).str.lower().str.contains(q, na=False)
        table = table[mask]

    # Ordre & PII
    cols = ["candidate_id","Nom","Localisation","CV_location","Source","Date candidature","Stage","Statut","Décision","Rejeté le","Raison rejet","LinkedIn"]
    if show_pii:
        cols += ["Email","Téléphone"]
    return table[cols].reset_index(drop=True)

# =========================
#   RENDER
# =========================
def render_job_card(row: pd.Series):
    title = row.get("title") or f"Job {row.get('job_id')}"
    desc = row.get("description", "")
    short, truncated = shorten_md(desc, max_chars=600)
    applicants = int(row.get("applicants_count", 0))

    st.markdown(f"### {title}")
    meta1, meta2 = st.columns([3, 2])
    with meta1:
        st.caption(f"ID: `{row.get('job_id')}` | Status: `{row.get('status')}` ")
    with meta2:
        st.caption(f"Créé: {row.get('created_at') or '—'} | Maj: {row.get('updated_at') or '—'}")

    st.metric("Candidatures", value=applicants)

    if truncated:
        with st.expander("Voir la description complète"):
            st.markdown(desc if desc else "_(pas de description)_")
    else:
        st.markdown(short if short else "_(pas de description)_")

    # ---- Bloc Candidats (INTERACTIF) ----
    with st.expander(f"👥 Candidats — {applicants}"):
        c1, c2 = st.columns([2, 1])
        with c1:
            local_q = st.text_input(f"Filtrer les candidats (job {row.get('job_id')})", "", key=f"cand_filter_{row.get('job_id')}")
        with c2:
            st.caption("Filtre par nom, localisation, source, stage, statut…")

        table = build_candidates_table_for_job(row.get("job_id"), local_filter=local_q)
        st.dataframe(table, use_container_width=True, hide_index=True)

        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger la liste (.csv)",
            data=csv,
            file_name=f"job_{row.get('job_id')}_candidats.csv",
            mime="text/csv",
            key=f"dl_{row.get('job_id')}"
        )

    st.divider()

# Affichage en grille
cards_per_row = max(1, int(per_row))
rows = [df_jobs_view.iloc[i:i+cards_per_row] for i in range(0, len(df_jobs_view), cards_per_row)]
for chunk in rows:
    cols = st.columns(len(chunk))
    for c, (_, r) in zip(cols, chunk.iterrows()):
        with c:
            render_job_card(r)

# =========================
#   DEBUG / DIAGNOSTIC
# =========================
with st.expander("🔬 Diagnostic (schémas détectés)"):
    st.write("**Colonnes jobs**:", list(df_jobs.columns))
    st.write("**Exemple job**:"); st.json(df_jobs.head(1).to_dict(orient="records"))

    st.write("**Colonnes job applications**:", list(df_apps.columns))
    st.dataframe(df_apps.head(10))

    st.write("**Colonnes candidates**:", list(df_cands.columns))
    st.dataframe(df_cands[["cv.profile.summary","cv.profile.location"]])

st.caption("ℹ️ Cache 5 min. POC : lecture blobs Silver uniquement (JSONL).")
