# … (imports et tout le haut de ton script inchangés) …
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from azure.storage.blob import BlobServiceClient

st.set_page_config(page_title="CV Compatibility — Jobs publics", page_icon="🧩", layout="wide")
st.title("🧩 CV Compatibility — Jobs publics (POC Streamlit)")
st.caption("Lecture directe des fichiers *silver* (JSONL) depuis Azure Blob Storage, sans appel API Teamtailor.")

# === FIX CSS: forcer l'affichage horizontal + ellipse des labels de boutons ===
st.markdown("""
<style>
/* Tous les boutons Streamlit */
.stButton > button {
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  min-width: 160px;               /* largeur mini pour éviter les retours à la ligne */
  height: 44px;                    /* hauteur homogène */
  border-radius: 14px;
}
/* Boutons de pagination ◀ ▶ */
div[data-testid="stHorizontalBlock"] .stButton > button {
  min-width: 64px;
}
</style>
""", unsafe_allow_html=True)

# =========================
#  Helpers & normalisation (identiques à ta version précédente)
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def get_blob_bytes(connection_string: str, container: str, blob_path: str) -> bytes:
    bsc = BlobServiceClient.from_connection_string(connection_string)
    return bsc.get_container_client(container).get_blob_client(blob_path).download_blob().readall()

def _coerce_json(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list): return obj
    if isinstance(obj, dict): return [obj]
    text = obj.decode("utf-8", errors="ignore") if isinstance(obj, (bytes, bytearray)) else (obj or "")
    if not isinstance(text, str): return []
    try:
        return _coerce_json(json.loads(text))
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
        return rows

@st.cache_data(show_spinner=True, ttl=300)
def load_json_from_blob(connection_string: str, container: str, blob_path: str) -> List[Dict[str, Any]]:
    return _coerce_json(get_blob_bytes(connection_string, container, blob_path))

def pick(row: pd.Series, key: str, default=None):
    return row[key] if key in row and pd.notna(row[key]) else default

def to_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)): return ""
    return str(val)

def join_nonempty(values: List[Any], sep=" ") -> str:
    return sep.join([to_str(v).strip() for v in values if to_str(v).strip()])

def csvify_list(val) -> str:
    if isinstance(val, list): return ", ".join(map(to_str, val))
    return to_str(val)

def normalize_jobs(jobs_raw: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(jobs_raw, sep=".")
    df["job_id"]      = df.apply(lambda r: pick(r, "job_id"), axis=1)
    df["title"]       = df.apply(lambda r: pick(r, "title"), axis=1)
    df["status"]      = df.apply(lambda r: pick(r, "status", ""), axis=1)
    df["description"] = df.apply(lambda r: pick(r, "body_text", ""), axis=1)
    df["locations"]   = df.apply(lambda r: csvify_list(pick(r, "locations", [])), axis=1)
    df["tags"]        = df.apply(lambda r: csvify_list(pick(r, "tags", [])), axis=1)
    df["department"]  = df.apply(lambda r: pick(r, "department", ""), axis=1)
    df["division"]    = df.apply(lambda r: pick(r, "division", ""), axis=1)
    df["created_at"]  = df.apply(lambda r: pick(r, "created_at", ""), axis=1)
    df["updated_at"]  = df.apply(lambda r: pick(r, "updated_at", ""), axis=1)
    keep = ["job_id","title","status","description","locations","tags","department","division","created_at","updated_at"]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]

def normalize_apps(apps_raw: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.json_normalize(apps_raw, sep=".")
    df["application_id"]       = df.apply(lambda r: pick(r, "application_id"), axis=1)
    df["job_id"]               = df.apply(lambda r: pick(r, "job_id"), axis=1)
    df["candidate_id"]         = df.apply(lambda r: pick(r, "candidate_id"), axis=1)
    df["created_at"]           = df.apply(lambda r: pick(r, "created_at"), axis=1)
    df["updated_at"]           = df.apply(lambda r: pick(r, "updated_at"), axis=1)
    df["stage_name"]           = df.apply(lambda r: pick(r, "stage_name"), axis=1)
    df["status"]               = df.apply(lambda r: pick(r, "status"), axis=1)
    df["decision"]             = df.apply(lambda r: pick(r, "decision"), axis=1)
    df["rejected_at"]          = df.apply(lambda r: pick(r, "rejected_at"), axis=1)
    df["reject_reason"]        = df.apply(lambda r: pick(r, "reject_reason_text"), axis=1)
    df["source_site"]          = df.apply(lambda r: pick(r, "source_site"), axis=1)
    df["source_url"]           = df.apply(lambda r: pick(r, "source_url"), axis=1)
    df["sourced"]              = df.apply(lambda r: pick(r, "sourced"), axis=1)
    df["changed_stage_at"]     = df.apply(lambda r: pick(r, "changed_stage_at"), axis=1)
    df["cover_letter_present"] = df.apply(lambda r: pick(r, "cover_letter_present"), axis=1)
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
    df["full_name"]    = [join_nonempty([f,l]) for f,l in zip(df["first_name"], df["last_name"])]
    df["email"]        = df.apply(lambda r: pick(r, "email"), axis=1)
    df["phone"]        = df.apply(lambda r: pick(r, "phone"), axis=1)
    df["location"]     = df.apply(lambda r: pick(r, "location"), axis=1)
    df["linkedin_url"] = df.apply(lambda r: pick(r, "linkedin_url"), axis=1)
    df["picture_url"]  = df.apply(lambda r: pick(r, "picture_url"), axis=1)
    df["created_at"]   = df.apply(lambda r: pick(r, "created_at"), axis=1)
    df["updated_at"]   = df.apply(lambda r: pick(r, "updated_at"), axis=1)
    df["source_site"]  = df.apply(lambda r: pick(r, "source_site"), axis=1)
    df["referring_url"]= df.apply(lambda r: pick(r, "referring_url"), axis=1)
    df["internal"]     = df.apply(lambda r: pick(r, "internal"), axis=1)
    df["sourced"]      = df.apply(lambda r: pick(r, "sourced"), axis=1)
    df["unsubscribed"] = df.apply(lambda r: pick(r, "unsubscribed"), axis=1)
    keep = ["candidate_id","full_name","location","email","phone","linkedin_url","picture_url","created_at","updated_at","source_site","referring_url","internal","sourced","unsubscribed"]
    extra = [c for c in df.columns if c not in keep]
    return df[keep + extra]

def is_public_row(row: pd.Series) -> bool:
    s = str(row.get("status", "") or "").lower()
    return any(k in s for k in ["public","published","open","active"])

def shorten_md(md: str, max_chars: int = 600) -> Tuple[str, bool]:
    if md is None: return "", False
    text = str(md)
    return (text, False) if len(text) <= max_chars else (text[:max_chars].rstrip()+"…", True)

def decision_icon(decision: Optional[str], status: Optional[str], rejected_at: Optional[str]) -> Tuple[str, str]:
    d = (decision or "").lower().strip()
    s = (status or "").lower().strip()
    if d in {"hired","accepted","offer_accepted","offer-accepted","approved","yes"} or s in {"accepted","hired","offer_accepted"}:
        return "🟢","Accepté"
    if d in {"rejected","declined","no"} or (rejected_at and str(rejected_at).strip()) or "reject" in s:
        return "🔴","Refusé"
    return "🟠","En attente"

# NEW: forcer le nom sur une seule ligne + troncature
def one_line_name(name: str, max_chars: int = 24) -> str:
    s = " ".join((name or "").split())      # nettoie espaces multiples
    if len(s) > max_chars:
        s = s[:max_chars-1] + "…"
    return s.replace(" ", "\u00A0")         # espaces insécables

# =========================
#   Sidebar & chargement
# =========================
st.sidebar.header("⚙️ Configuration Blob (lecture)")
connection_string = st.sidebar.text_input("Connection string Azure","DefaultEndpointsProtocol=https;AccountName=absaifabrik;AccountKey=dHnt6TqjK6GYHvEjLTKenFdRHitSCByxcqenpCAvP/+GkY6XjHk7+BMfSpVuhbSpUi/5EfGq61CR+AStw0NiCA==;EndpointSuffix=core.windows.net" , type="password")
container_name    = st.sidebar.text_input("Nom du container", "cvcompat")

st.sidebar.subheader("📄 Fichiers Silver (JSONL)")
blob_jobs       = st.sidebar.text_input("Jobs", "silver/jobs/2025-08-29T08-23-25Z.jsonl")
blob_apps       = st.sidebar.text_input("Job applications", "silver/job-applications/2025-09-01T08-34-48Z.jsonl")
blob_candidates = st.sidebar.text_input("Candidates unifié", "silver/candidates_unified/2025-09-01T09-31-47Z.jsonl")

with st.sidebar.expander("🔎 Filtres / Affichage", expanded=True):
    search_query = st.text_input("Recherche dans titre/description", "")
    sort_by      = st.selectbox("Trier par", ["Titre (A→Z)", "Nb candidatures (desc)", "Nb candidatures (asc)"])
    per_row      = st.slider("Cartes job par ligne", 1, 4, 2)
    page_size    = st.slider("Candidats par page (défilé)", 6, 40, 12, step=2)
    pills_per_row= st.slider("Pills par ligne", 3, 8, 4)   # <— NOUVEAU : pour la largeur des noms
    show_pii     = st.checkbox("Afficher e-mail / téléphone (PII)", value=False)

if not connection_string or not container_name:
    st.info("➡️ Renseigne la **connection string** et le **container**.")
    st.stop()

try:
    jobs_raw = load_json_from_blob(connection_string, container_name, blob_jobs)
    apps_raw = load_json_from_blob(connection_string, container_name, blob_apps)
    cands_raw= load_json_from_blob(connection_string, container_name, blob_candidates)
except Exception as e:
    st.error(f"Échec de lecture des blobs. Détails: {e}")
    st.stop()

df_jobs = normalize_jobs(jobs_raw)
df_apps = normalize_apps(apps_raw)
df_cands= normalize_candidates(cands_raw)

df_jobs_public = df_jobs[df_jobs.apply(is_public_row, axis=1)].copy()
apps_count = (df_apps.dropna(subset=["job_id"]).groupby("job_id").size().rename("applicants_count").reset_index())
df_jobs_view = df_jobs_public.merge(apps_count, on="job_id", how="left")
df_jobs_view["applicants_count"] = df_jobs_view["applicants_count"].fillna(0).astype(int)

if search_query.strip():
    q = search_query.lower().strip()
    mask = df_jobs_view["title"].astype(str).str.lower().str.contains(q, na=False) | \
           df_jobs_view["description"].astype(str).str.lower().str.contains(q, na=False)
    df_jobs_view = df_jobs_view[mask]

if sort_by == "Titre (A→Z)":
    df_jobs_view = df_jobs_view.sort_values(by=["title","job_id"])
elif sort_by == "Nb candidatures (desc)":
    df_jobs_view = df_jobs_view.sort_values(by=["applicants_count","title"], ascending=[False, True])
else:
    df_jobs_view = df_jobs_view.sort_values(by=["applicants_count","title"], ascending=[True, True])

st.success(f"✅ {len(df_jobs_view)} job(s) public(s) chargé(s).")

# =========================
#   Candidats: pills & détail
# =========================
def get_candidates_for_job(job_id: Any) -> pd.DataFrame:
    apps = df_apps[df_apps["job_id"].astype(str) == str(job_id)].copy()
    if apps.empty:
        return pd.DataFrame(columns=["candidate_id","full_name","decision","status","rejected_at","stage_name",
                                     "created_at","location","linkedin_url","email","phone","source_site","picture_url","reject_reason"])
    merged = apps.merge(df_cands, on="candidate_id", how="left", suffixes=("", "_cand"))
    out = pd.DataFrame({
        "candidate_id": merged["candidate_id"],
        "full_name": merged["full_name"],
        "decision": merged["decision"],
        "status": merged["status"],
        "rejected_at": merged["rejected_at"],
        "stage_name": merged["stage_name"],
        "created_at": merged["created_at"],
        "location": merged["location"],
        "linkedin_url": merged["linkedin_url"],
        "email": merged["email"],
        "phone": merged["phone"],
        "source_site": merged["source_site"],
        "picture_url": merged.get("picture_url", None),
        "reject_reason": merged.get("reject_reason", None),
    }).sort_values(by="created_at", ascending=False, na_position="last").reset_index(drop=True)
    return out

def render_candidate_detail(row: pd.Series):
    emoji, label = decision_icon(row.get("decision"), row.get("status"), row.get("rejected_at"))
    st.markdown(f"#### {row.get('full_name') or row.get('candidate_id')}")
    st.caption(f"{emoji} {label} • Stage: {row.get('stage_name') or '—'} • Statut: {row.get('status') or '—'} • Candidature: {row.get('created_at') or '—'}")
    c1, c2 = st.columns([2,1])
    with c1:
        st.markdown(f"**Localisation** : {row.get('location') or '—'}")
        st.markdown(f"**Source** : {row.get('source_site') or '—'}")
        if row.get("linkedin_url"): st.link_button("Profil LinkedIn", row.get("linkedin_url"))
        if row.get("reject_reason"): st.markdown(f"**Raison de rejet** : {row.get('reject_reason')}")
    with c2:
        if row.get("picture_url"): st.image(row.get("picture_url"), caption="Photo", use_column_width=True)
        if show_pii:
            st.markdown(f"**Email** : {row.get('email') or '—'}")
            st.markdown(f"**Téléphone** : {row.get('phone') or '—'}")

def render_job_card(row: pd.Series):
    job_id = row.get("job_id"); title = row.get("title") or f"Job {job_id}"
    desc = row.get("description", "")
    short, truncated = shorten_md(desc, max_chars=600)
    applicants = int(row.get("applicants_count", 0))

    st.markdown(f"### {title}")
    meta1, meta2 = st.columns([3,2])
    with meta1:
        st.caption(f"ID: `{job_id}` | Status: `{row.get('status')}` | Lieux: {row.get('locations') or '—'} | Tags: {row.get('tags') or '—'}")
    with meta2:
        st.caption(f"Dept/Div: {row.get('department') or '—'} / {row.get('division') or '—'} | Créé: {row.get('created_at') or '—'} | Maj: {row.get('updated_at') or '—'}")
    st.metric("Candidatures", value=applicants)

    if truncated:
        with st.expander("Voir la description complète"): st.markdown(desc if desc else "_(pas de description)_")
    else:
        st.markdown(short if short else "_(pas de description)_")

    with st.expander(f"👥 Candidats — {applicants}"):
        cands = get_candidates_for_job(job_id)

        page_key = f"page_{job_id}"
        sel_key  = f"selected_cand_{job_id}"
        if page_key not in st.session_state: st.session_state[page_key] = 0
        if sel_key not in st.session_state:  st.session_state[sel_key] = None

        total = len(cands)
        if total == 0:
            st.info("Aucune candidature pour cette offre.")
        else:
            max_page = max(0, (total - 1) // page_size)
            nav1, nav2, nav3 = st.columns([1,2,1])
            with nav1:
                if st.button("◀", key=f"prev_{job_id}", use_container_width=True) and st.session_state[page_key] > 0:
                    st.session_state[page_key] -= 1
            with nav2:
                st.caption(f"Page {st.session_state[page_key] + 1} / {max_page + 1} — {total} candidats")
            with nav3:
                if st.button("▶", key=f"next_{job_id}", use_container_width=True) and st.session_state[page_key] < max_page:
                    st.session_state[page_key] += 1

            start = st.session_state[page_key] * page_size
            end   = min(start + page_size, total)
            window = cands.iloc[start:end]

            # ——— Chips/pills sur une ou plusieurs rangées ———
            cols_per_row = int(pills_per_row)  # <— contrôlable dans la sidebar
            rows_needed = (len(window) + cols_per_row - 1) // cols_per_row
            idx = 0
            for _ in range(rows_needed):
                cols = st.columns(cols_per_row)
                for col in cols:
                    if idx >= len(window): break
                    wrow = window.iloc[idx]; idx += 1
                    emoji, _ = decision_icon(wrow.get("decision"), wrow.get("status"), wrow.get("rejected_at"))
                    label = f"{emoji} {one_line_name(wrow.get('full_name') or str(wrow.get('candidate_id')))}"
                    with col:
                        if st.button(label, key=f"btn_{job_id}_{wrow.get('candidate_id')}", use_container_width=True):
                            st.session_state[sel_key] = wrow.get("candidate_id")

            if st.session_state[sel_key]:
                selected = cands[cands["candidate_id"] == st.session_state[sel_key]]
                if not selected.empty:
                    st.markdown("---")
                    render_candidate_detail(selected.iloc[0])
                else:
                    st.warning("Candidat introuvable dans la page courante.")

    st.divider()

# ——— Affichage des jobs en grille ———
cards_per_row = max(1, int(per_row))
rows = [df_jobs_view.iloc[i:i+cards_per_row] for i in range(0, len(df_jobs_view), cards_per_row)]
for chunk in rows:
    cols = st.columns(len(chunk))
    for c, (_, r) in zip(cols, chunk.iterrows()):
        with c: render_job_card(r)

# Debug (optionnel)
with st.expander("🔬 Diagnostic (schémas détectés)"):
    st.write("**Colonnes jobs**:", list(df_jobs.columns)); st.json(df_jobs.head(1).to_dict(orient="records"))
    st.write("**Colonnes job applications**:", list(df_apps.columns)); st.dataframe(df_apps.head(10))
    st.write("**Colonnes candidates**:", list(df_cands.columns)); st.dataframe(df_cands[["candidate_id","full_name","location","email","phone","linkedin_url","source_site"]].head(10))

st.caption("ℹ️ Cache 5 min. POC : lecture blobs Silver uniquement (JSONL).")
