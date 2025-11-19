import os
import json
import math
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

# Désactiver les warnings TensorFlow au démarrage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import joblib


# Optionnels mais utiles pour /score_job (lecture JSONL local)
try:
    import pandas as pd
except Exception:
    pd = None

# Optionnel : TF-IDF cosine
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# =========================
# Config
# =========================
# Chercher les modèles dans results_TIMESTAMP/models/ (généré par le notebook)
# ou dans model_results/models/ (legacy)
def find_model_dir():
    """
    Cherche le répertoire des modèles dans cet ordre:
    1. Variable d'environnement MODEL_DIR
    2. Répertoire results_*/models/ le plus récent (généré par le notebook)
    3. Répertoire model_results/models/ (legacy)
    """
    # 1. Vérifier variable d'environnement
    if os.getenv("MODEL_DIR"):
        return os.getenv("MODEL_DIR")

    # 2. Chercher results_*/models/ le plus récent
    api_dir = os.path.dirname(__file__)
    results_dirs = []
    try:
        for item in os.listdir(api_dir):
            if item.startswith("results_") and os.path.isdir(os.path.join(api_dir, item)):
                models_subdir = os.path.join(api_dir, item, "models")
                if os.path.isdir(models_subdir):
                    results_dirs.append((item, models_subdir))

        if results_dirs:
            # Trier par nom (timestamp) et prendre le plus récent
            results_dirs.sort(reverse=True)
            latest_dir = results_dirs[0][1]
            print(f"✅ Found results directory: {results_dirs[0][0]}/models/")
            return latest_dir
    except Exception as e:
        print(f"⚠️  Error searching for results_* directories: {e}")

    # 3. Fallback à model_results/models/
    default_dir = os.path.join(api_dir, "model_results", "models")
    print(f"Using default directory: model_results/models/")
    return default_dir

MODEL_DIR = find_model_dir()
DATA_PATH_DEFAULT = os.getenv("APPLICATIONS_GOLD_PATH", "applications_gold_latest-8.jsonl")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "intfloat/multilingual-e5-base")

# Liste d'écoles "target" (csv) pour la feature 'ecole'
ECOLES_ENV = os.getenv("ECOLES_NAMES", "")
TARGET_ECOLES = [x.strip().lower() for x in ECOLES_ENV.split(",") if x.strip()]

# Types canoniques (doivent correspondre aux clés de TYPE_PACKS)
JOB_TYPE_CANON = [
    "data_engineer",
    "data_scientist",
    "ml_engineer",
    "data_analyst",
    "backend_engineer",
    "frontend_engineer",
    "product_manager",
]

# Règles rapides (mots-clés)
JOB_TYPE_RULES = {
    "data_engineer": ["data engineer", "ingénieur data", "pipeline", "etl", "elt", "spark", "airflow", "databricks", "lakehouse"],
    "data_scientist": ["data scientist", "nlp", "modélisation", "xgboost", "classification", "régression", "métriques"],
    "ml_engineer": ["ml engineer", "ml ops", "mlops", "serving", "monitoring", "feature store", "inference"],
    "data_analyst": ["data analyst", "power bi", "tableau", "dashboard", "reporting", "kpi"],
    "backend_engineer": ["backend", "api", "microservices", "fastapi", "django", "spring", "java"],
    "frontend_engineer": ["frontend", "react", "vue", "angular", "ui", "ux"],
    "product_manager": ["product manager", "chef de produit", "roadmap", "priorisation", "discovery"],
}

# Prototypes texte (pour détection embeddings)
JOB_TYPE_PROTOTYPES = {
    "data_engineer": "Design and build data pipelines, ETL/ELT, Spark, Airflow, data lakes, data warehouses.",
    "data_scientist": "Statistical modeling, NLP, classification/regression, experimentation, XGBoost, scikit-learn.",
    "ml_engineer": "Deploy ML models to production, MLOps, CI/CD for ML, feature store, monitoring, scaling.",
    "data_analyst": "Dashboards, Power BI/Tableau, SQL analytics, KPIs, reporting, business insights.",
    "backend_engineer": "Build APIs and services, scalability, microservices, Python/Java, FastAPI/Django/Spring.",
    "frontend_engineer": "Build web UIs with React/Vue/Angular, components, UX, state management.",
    "product_manager": "Own product roadmap, discovery, prioritization, stakeholder alignment, user stories.",
}

from contextlib import asynccontextmanager
import threading

# Flag pour indiquer que l'API est prête
_api_ready = False
_api_ready_lock = threading.Lock()

def _load_embedder_eager():
    """Charge le modèle d'embeddings de manière synchrone au démarrage"""
    import time

    print("\n⏳ Chargement du modèle d'embeddings (intfloat/multilingual-e5-base)...")
    print("   ⚠️  Ceci peut prendre 30-60 secondes...")
    print("   📥 Téléchargement et initialisation du modèle en cours...\n")

    try:
        from sentence_transformers import SentenceTransformer

        global _embedder
        start_time = time.time()

        # Charger le modèle (utilise le device par défaut, comme dans le notebook)
        print(f"   🔄 Importation du modèle SentenceTransformer...")
        _embedder = SentenceTransformer(EMB_MODEL_NAME)

        elapsed = time.time() - start_time
        print(f"\n✅ Modèle d'embeddings chargé avec succès!")
        print(f"   ⏱️  Temps total: {elapsed:.1f} secondes")
        print(f"   📊 Modèle: {EMB_MODEL_NAME}")
        print(f"   💾 Taille: ~500MB\n")
        return True
    except Exception as e:
        print(f"\n❌ Erreur lors du chargement du modèle d'embeddings:")
        print(f"   Détails: {type(e).__name__}: {str(e)[:500]}\n")
        import traceback
        traceback.print_exc()
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application (startup/shutdown)"""
    global _api_ready

    # Startup
    print("\n" + "="*80)
    print("🚀 DÉMARRAGE DE L'API CV COMPATIBILITY - Chargement des ressources...")
    print("="*80)

    try:
        # Charger le modèle d'embeddings MAINTENANT
        print("\n📋 Ressources à charger:")
        print("   1. Modèles XGBoost (global + spécialisés)... ✅ (déjà chargés)")
        print("   2. TF-IDF vectorizer... ✅ (déjà chargé)")
        print("   3. Job statistics... ✅ (déjà chargées)")
        print("   4. Modèle d'embeddings E5... ⏳ (en cours)\n")

        success = _load_embedder_eager()

        with _api_ready_lock:
            _api_ready = True

        if success:
            print("\n" + "="*80)
            print("✅ API PRÊTE - Tous les modèles chargés avec succès!")
            print("="*80)
            print("\n📊 État final:")
            print("   • Modèles XGBoost: ✅ Opérationnel")
            print("   • Embeddings E5: ✅ Opérationnel")
            print("   • TF-IDF: ✅ Opérationnel")
            print("   • Job Stats: ✅ Opérationnel")
            print("\n🌐 API accessible à: http://localhost:8000")
            print("📖 Documentation: http://localhost:8000/docs\n")
        else:
            print("\n" + "="*80)
            print("❌ ERREUR: Modèle d'embeddings n'a pas pu être chargé")
            print("="*80)
            print("   Le modèle d'embeddings est OBLIGATOIRE pour fonctionner.")
            print("   Vérifiez les erreurs ci-dessus et relancez l'API.\n")

    except Exception as e:
        print(f"\n❌ ERREUR lors du démarrage: {e}")
        with _api_ready_lock:
            _api_ready = True

    yield

    # Shutdown (optionnel)
    print("\n" + "="*80)
    print("🛑 Arrêt de l'API...")

app = FastAPI(title="CV Compatibility API", version="2.1", lifespan=lifespan)

# =========================
# Artéfacts (chargement)
# =========================
def safe_load(path: str) -> Optional[Any]:
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"⚠️  WARNING: Could not load {path}: {type(e).__name__}: {str(e)[:200]}")
        return None

GLOBAL_PACK = safe_load(os.path.join(MODEL_DIR, "global_pack.pkl"))   # {"cal": CalibratedModel, "cutoff": float}
TYPE_PACKS  = safe_load(os.path.join(MODEL_DIR, "type_packs.pkl"))    # {type: {"cal":..., "cutoff":...}}
CAND_FEATS  = safe_load(os.path.join(MODEL_DIR, "features.pkl"))      # list[str] (ordre des features)
ALPHA_BY_TYPE = safe_load(os.path.join(MODEL_DIR, "alpha_by_type.pkl")) or {}
TFIDF = safe_load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))    # peut être None

# Classif optionnel (non utilisé - le notebook utilise infer_job_type_row interne)
# JOB_TYPE_CLF et JOB_TYPE_LE ne sont pas sauvegardés dans le notebook
JOB_TYPE_CLF = None
JOB_TYPE_LE  = None

# Statistiques historiques des jobs (pour features 18-21)
JOB_STATS = safe_load(os.path.join(MODEL_DIR, "job_stats.pkl"))  # {"job_selectivity_historical": {...}, "job_competition_index_norm": {...}}

if TYPE_PACKS is None:
    TYPE_PACKS = {}
if JOB_STATS is None:
    JOB_STATS = {"job_selectivity_historical": {}, "job_competition_index_norm": {}}

# =========================
# Embeddings E5 (lazy, avec synchronisation)
# =========================
_embedder = None
_embedder_lock = threading.Lock()

def _hash_text(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8", "ignore")).hexdigest()

@lru_cache(maxsize=4096)
def _embed_cached(txt: str) -> np.ndarray:
    model = get_embedder()
    vec = model.encode(txt, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vec, dtype=np.float32)

def get_embedder():
    """Récupère le modèle d'embeddings (doit être chargé au démarrage)"""
    global _embedder

    if _embedder is None:
        # Cette erreur signifie que le modèle n'a pas pu être chargé au démarrage
        raise HTTPException(
            status_code=503,
            detail="Modèle d'embeddings non chargé au démarrage. Vérifiez les logs et réessayez."
        )

    return _embedder

def cos_sim_dense(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    va = _embed_cached(a)
    vb = _embed_cached(b)
    return float(np.dot(va, vb))  # embeddings normalisés → cos

def cos_sim_tfidf(a: str, b: str) -> float:
    if TFIDF is None or not SKLEARN_AVAILABLE:
        return 0.0
    if not a or not b:
        return 0.0
    try:
        X = TFIDF.transform([a, b])
        sim = cosine_similarity(X[0], X[1])
        return float(sim[0, 0])
    except Exception:
        return 0.0

# =========================
# Utils clean / parse
# =========================
def _norm_ws(s: Optional[str]) -> str:
    if not s:
        return ""
    return " ".join(str(s).split())

def _safe_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        return [x]
    return list(x)

def _tok_set(s: str) -> set:
    return set([t for t in "".join([c if c.isalnum() else " " for c in s.lower()]).split() if t])

def parse_years_from_text(s: str) -> int:
    import re
    s = (s or "").lower()
    patterns = [
        r"(\d+)\s*\+?\s*(ans|an|years|year)",
        r"au moins\s*(\d+)\s*(ans|an)",
        r"minimum\s*(\d+)\s*(ans|an)",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return 0

def get_job_text(job: Dict[str, Any]) -> Dict[str, str]:
    title = _norm_ws(job.get("job_title") or job.get("title") or "")
    desc = _norm_ws(job.get("job_description_text") or job.get("description") or "")
    seg = job.get("job_llm_segments") or {}
    if isinstance(seg, str):
        try:
            seg = json.loads(seg)
        except Exception:
            seg = {}
    responsibilities = _norm_ws(seg.get("responsibilities") or "")
    requirements = _norm_ws(seg.get("requirements") or seg.get("requirements_must") or "")
    job_text_full = " ".join([title, desc, responsibilities, requirements]).strip()
    return {
        "title": title,
        "desc": desc,
        "responsibilities": responsibilities,
        "requirements": requirements,
        "full": job_text_full
    }

def get_cv_text(cv: Dict[str, Any]) -> Dict[str, str]:
    exp = _norm_ws(cv.get("text_cv_experience") or "")
    skills = _norm_ws(cv.get("text_cv_skills") or "")
    full = _norm_ws(cv.get("text_cv_full") or "")
    cv_full_plus = " ".join([exp, skills, full]).strip()
    return {"exp": exp, "skills": skills, "full": full, "full_plus": cv_full_plus}

def coverage_ratio(required: List[str], provided_text: str) -> float:
    if not required:
        return 0.0
    toks = _tok_set(provided_text or "")
    req = [r.strip().lower() for r in required if str(r).strip()]
    if not req:
        return 0.0
    hit = 0
    for r in req:
        r_toks = set(r.split())
        if r_toks & toks:
            hit += 1
    return float(hit) / float(len(req))

def languages_coverage(job_langs: List[str], cv_langs: List[str]) -> float:
    jl = set([x.strip().lower() for x in job_langs if str(x).strip()])
    cl = set([x.strip().lower() for x in cv_langs if str(x).strip()])
    if not jl:
        return 0.0
    return float(len(jl & cl)) / float(len(jl))

def critical_skills_blocker(job_text: str, cv_text: str) -> int:
    jt = (job_text or "").lower()
    if any(k in jt for k in ["must have", "obligatoire", "required", "indispensable"]):
        sim = cos_sim_dense(f"query: {_norm_ws(job_text)}", f"passage: {_norm_ws(cv_text)}")
        return 1 if sim < 0.45 else 0
    return 0

def ecole_feature(cv_text: str) -> int:
    if not TARGET_ECOLES:
        return 0
    c = (cv_text or "").lower()
    for e in TARGET_ECOLES:
        if e and e in c:
            return 1
    return 0

def overqualification_penalty(cv_years: int, job_years: int) -> float:
    if job_years <= 0:
        return 0.0
    gap = cv_years - job_years
    if gap <= 0:
        return 0.0
    return float(min(0.2, 0.02 * gap))

def get_job_stats(job_id: Optional[str]) -> Tuple[float, float]:
    """
    Récupère les statistiques historiques d'une offre.

    Returns:
        (job_selectivity_historical, job_competition_index_norm)
        Valeurs par défaut: (0.5, 0.5) si job_id non trouvé
    """
    if not job_id or JOB_STATS is None:
        return 0.5, 0.5

    selectivity = float(
        JOB_STATS.get("job_selectivity_historical", {}).get(job_id, 0.5)
    )
    competition = float(
        JOB_STATS.get("job_competition_index_norm", {}).get(job_id, 0.5)
    )
    return selectivity, competition

# =========================
# Détection type d'offre
# =========================
@lru_cache(maxsize=1)
def _proto_embeds():
    emb = get_embedder()
    names, vecs = [], []
    for t, proto in JOB_TYPE_PROTOTYPES.items():
        v = emb.encode(f"query: {proto}", normalize_embeddings=True)
        names.append(t)
        vecs.append(v)
    return names, np.vstack(vecs).astype(np.float32)

def _rule_based_job_type(title: str, desc_or_full: str) -> Optional[str]:
    text = f"{title} {desc_or_full}".lower()
    for t, kws in JOB_TYPE_RULES.items():
        for kw in kws:
            if kw in text:
                return t
    return None

def _embed_based_job_type(title: str, full_text: str) -> Optional[str]:
    try:
        names, V = _proto_embeds()
        q = _embed_cached(f"query: {title or full_text}")
        sims = V @ q
        idx = int(np.argmax(sims))
        if float(sims[idx]) >= 0.35:
            return names[idx]
        q2 = _embed_cached(f"query: {full_text}")
        sims2 = V @ q2
        idx2 = int(np.argmax(sims2))
        return names[idx2]
    except Exception:
        return None

def detect_job_type(job: Dict[str, Any]) -> Optional[str]:
    # 0) si présent dans GOLD
    jt = job.get("job_type_small")
    if isinstance(jt, str) and jt.strip():
        return jt.strip()

    jtxt = get_job_text(job)
    title = jtxt["title"]
    full = jtxt["full"] or (jtxt["desc"] + " " + jtxt["responsibilities"] + " " + jtxt["requirements"])

    # 1) classif optionnelle (si fournie et TFIDF dispo)
    if JOB_TYPE_CLF is not None and JOB_TYPE_LE is not None and TFIDF is not None:
        try:
            X = TFIDF.transform([f"{title} {full}".strip()])
            y = JOB_TYPE_CLF.predict(X)[0]
            pred = JOB_TYPE_LE.inverse_transform([y])[0]
            if pred in JOB_TYPE_CANON:
                return pred
        except Exception:
            pass

    # 2) règles
    rb = _rule_based_job_type(title, full)
    if rb:
        return rb

    # 3) embeddings
    eb = _embed_based_job_type(title, full)
    return eb

# =========================
# Pydantic models
# =========================
from pydantic import RootModel

class JobDict(RootModel):
    root: Dict[str, Any]
    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        return self.root

class CVDict(RootModel):
    root: Dict[str, Any]
    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        return self.root

class OnePair(BaseModel):
    pair_id: Optional[str] = None
    job: Optional[JobDict] = None
    cv: Optional[CVDict] = None
    # Mode minimal alternatif
    job_text: Optional[str] = None
    cv_text: Optional[str] = None

class ManyPairs(BaseModel):
    items: List[OnePair] = Field(default_factory=list)

class ScoreOut(BaseModel):
    pair_id: Optional[str] = None
    score_global: Optional[float] = None
    score_type: Optional[float] = None
    score_final: float
    cutoff: float
    pred: int
    job_type_small: Optional[str] = None
    features_used: Optional[Dict[str, float]] = None  # optionnel

class ScoreJobOutItem(BaseModel):
    candidate_id: str
    score_global: Optional[float] = None
    score_type: Optional[float] = None
    score_final: float
    cutoff: float
    pred: int
    rank_in_offer_pool_norm: float
    features_used: Optional[Dict[str, float]] = None  # optionnel

class ScoreJobOut(BaseModel):
    job_id: str
    job_type_small: Optional[str] = None
    n_candidates: int
    items: List[ScoreJobOutItem]

# =========================
# Feature builder principal
# =========================
def build_features(job: Dict[str, Any], cv: Dict[str, Any]) -> Dict[str, float]:
    jtxt = get_job_text(job)
    ctxt = get_cv_text(cv)
    

    # Dense sims (E5, query/passages)
    sim_dense_full = cos_sim_dense(f"query: {jtxt['full']}", f"passage: {ctxt['full_plus']}")
    sim_exp_title = cos_sim_dense(f"query: {jtxt['title']}", f"passage: {ctxt['exp']}")
    sim_exp_responsibilities = cos_sim_dense(f"query: {jtxt['responsibilities']}", f"passage: {ctxt['exp']}")

    # Sparse sim (TF-IDF si dispo)
    sim_sparse_tfidf = cos_sim_tfidf(jtxt["full"], ctxt["full_plus"])

    # Skills coverage & extraction
    job_required_skills_plus = _safe_list(job.get("job_required_skills_plus"))
    cv_skills_text = ctxt["skills"] or ctxt["full_plus"]
    skill_gap_ratio = 1.0 - coverage_ratio(job_required_skills_plus, cv_skills_text)

    # skill_gap_per_year_required: compétences manquantes par année requise
    # Nombre de skills manquantes = job_required_skills - (job_required_skills ∩ cv_skills)
    job_req_set = set([s.lower().strip() for s in job_required_skills_plus if s])
    cv_tok_set = _tok_set(cv_skills_text)
    skill_job_only = len([s for s in job_req_set if not any(t in cv_tok_set for t in s.split())])
    job_years = job.get("job_required_years_num")
    if job_years is None:
        job_years = parse_years_from_text(jtxt["requirements"] or jtxt["desc"])
    job_years = max(1, int(job_years or 1))  # Éviter division par zéro
    skill_gap_per_year_required = float(skill_job_only) / float(job_years)

    # Langues
    lang_required_cov = languages_coverage(
        _safe_list(job.get("job_languages_required")),
        _safe_list(cv.get("cv_languages")),
    )

    # Années / seniority
    cv_years = cv.get("cv_years_experience_num")
    if cv_years is None:
        cv_years = parse_years_from_text(ctxt["exp"] or ctxt["full_plus"])
    exp_gap_years = int(cv_years) - int(job_years)
    seniority_gap = float(exp_gap_years)

    # Soft skills (liste courte)
    soft_list = ["communication", "team", "autonomie", "organisé", "rigoureux",
                 "leadership", "curiosité", "adaptabilité", "proactivité", "empathie"]
    cov_soft_skills = coverage_ratio(soft_list, cv_skills_text)

    # Blockers
    blocker = critical_skills_blocker(jtxt["requirements"] or jtxt["desc"], cv_skills_text)

    # Ecole
    ecole = ecole_feature(ctxt["full_plus"])

    # Surqualification
    overq = overqualification_penalty(cv_years, job_years)

    # sim_x_seniority: interaction entre similarité et niveau de seniority
    # Favorise les candidats avec bon match ET bon niveau
    seniority_rank = 1.0 + max(0.0, min(5.0, seniority_gap))  # Normalise entre 1-6
    sim_x_seniority = float(sim_dense_full) * seniority_rank

    # soft_skills_weighted: combinaison pondérée soft skills + école (proxy du fit culturel)
    # Formule: cov_soft_skills * 0.5 + ecole * 0.3 + (1 - skill_gap_ratio) * 0.2
    soft_skills_weighted = float(
        cov_soft_skills * 0.5 +
        ecole * 0.3 +
        (1.0 - skill_gap_ratio) * 0.2
    )

    # Features de sélectivité (chargées depuis job_stats.pkl)
    job_id = job.get("job_id")
    job_selectivity_historical, job_competition_index_norm = get_job_stats(job_id)

    # is_low_selectivity_boost: boost pour bons candidats sur offres peu sélectives
    # Objectif: réduire les faux négatifs sur offres avec peu d'acceptations
    # Formule: (ecole + rank) × (1 si selectivity < 0.3 else 0)
    # NOTE: rank_in_offer_pool_norm sera 0.5 ici (valeur par défaut), remplacé en /score_job
    is_low_selectivity_boost = float(
        (ecole + 0.5) * (1.0 if job_selectivity_historical < 0.3 else 0.0)
    )

    # job_candidate_mismatch_flags: pénalité pour mismatches critiques
    # Objectif: bloquer candidats avec incompatibilités évidentes (anti-FP)
    # 3 flags:
    # - flag1: pas d'école cible ET offre très sélective (>0.6)
    # - flag2: très gros gap de compétences (ratio < -2)
    # - flag3: très surqualifié (écart > 5 ans)
    flag_no_school_selective = (ecole == 0) and (job_selectivity_historical > 0.6)
    flag_big_skill_gap = skill_gap_ratio < -2.0  # Note: skill_gap_ratio est (1 - coverage), donc < -2 signifie coverage > 3
    flag_overqualified = exp_gap_years > 5

    num_flags = int(flag_no_school_selective) + int(flag_big_skill_gap) + int(flag_overqualified)
    job_candidate_mismatch_flags = float(-1.0 * num_flags)

    # ORDRE DES FEATURES (doit correspondre à l'entraînement XGBoost du notebook)
    # Phase 5: Sélection de Features Finales (Cellule 21)
    feats = {
        # 1-4. Similarité (4 features)
        "sim_dense_full": float(sim_dense_full),
        "sim_sparse_tfidf": float(sim_sparse_tfidf),
        "sim_exp_title": float(sim_exp_title),
        "sim_exp_responsibilities": float(sim_exp_responsibilities),

        # 5-8. Skills (4 features)
        "skill_gap_ratio": float(skill_gap_ratio),
        "skill_gap_per_year_required": float(skill_gap_per_year_required),
        "cov_soft_skills": float(cov_soft_skills),
        "critical_skills_blocker": float(blocker),

        # 9-11. Expérience (3 features)
        "job_required_years_num": float(job_years),
        "exp_gap_years": float(exp_gap_years),
        "overqualification_penalty": float(overq),

        # 12. Seniority (1 feature)
        "seniority_gap": float(seniority_gap),

        # 13. Langues (1 feature)
        "lang_required_coverage": float(lang_required_cov),

        # 14. École (1 feature)
        "ecole": float(ecole),

        # 15-16. Composites (2 features)
        "sim_x_seniority": float(sim_x_seniority),
        "soft_skills_weighted": float(soft_skills_weighted),

        # 17. Positionnement (1 feature)
        "rank_in_offer_pool_norm": 0.5,  # Valeur par défaut (moyen), remplacée en /score_job

        # 18-19. Sélectivité (2 features) - Chargées depuis job_stats.pkl
        "job_selectivity_historical": float(job_selectivity_historical),
        "job_competition_index_norm": float(job_competition_index_norm),

        # 20-21. Phase 4J - Features non-linéaires (2 features)
        "is_low_selectivity_boost": float(is_low_selectivity_boost),
        "job_candidate_mismatch_flags": float(job_candidate_mismatch_flags),
    }

    # Propager job_type_small si fourni
    if "job_type_small" in job and isinstance(job["job_type_small"], str):
        feats["_job_type_small"] = job["job_type_small"]

    return feats

def align_feature_vector(feats_dict: Dict[str, float], feat_order: Optional[List[str]]) -> np.ndarray:
    if not feat_order:
        keys = sorted([k for k in feats_dict.keys() if not k.startswith("_")])
        return np.array([feats_dict[k] for k in keys], dtype=float)
    return np.array([float(feats_dict.get(f, 0.0)) for f in feat_order], dtype=float)

def get_job_type(job: Dict[str, Any], feats_dict: Dict[str, float], override: Optional[str] = None) -> Optional[str]:
    if override:
        return override
    jt = job.get("job_type_small")
    if isinstance(jt, str) and jt.strip():
        return jt.strip()
    return detect_job_type(job)

def normalize_score(score: float, method: str = "sigmoid") -> float:
    """
    Normalise un score pour mieux distribuer les valeurs.

    Args:
        score: Score brut (0-1)
        method: Méthode de normalisation
            - "sigmoid": Transformation sigmoïde centrée sur 0.6 (étale 50-70% → 0-100%)
            - "linear": Étalement linéaire (mapping 0.4-0.8 → 0-1)
            - "percentile": Transformation basée sur une distribution cible

    Returns:
        Score normalisé (0-1)
    """
    if method == "sigmoid":
        # Sigmoïde centrée sur 0.6 avec forte pente
        # Étale les scores autour de 0.5-0.7 sur toute la plage
        # Formule: 1 / (1 + exp(-k * (x - center)))
        center = 0.6
        steepness = 15.0  # Plus élevé = plus d'étalement
        normalized = 1.0 / (1.0 + math.exp(-steepness * (score - center)))
        return float(np.clip(normalized, 0.0, 1.0))

    elif method == "linear":
        # Mapping linéaire: [0.4, 0.8] → [0.0, 1.0]
        # Scores en dehors de cette plage sont clippés
        min_score = 0.4
        max_score = 0.8
        normalized = (score - min_score) / (max_score - min_score)
        return float(np.clip(normalized, 0.0, 1.0))

    elif method == "percentile":
        # Transformation basée sur une distribution normale
        # Centre à 0.6, écart-type 0.1
        # Convertit en percentile de distribution normale
        mean = 0.6
        std = 0.1
        z_score = (score - mean) / std
        # Fonction de répartition normale (CDF)
        from scipy import special
        try:
            normalized = 0.5 * (1.0 + special.erf(z_score / math.sqrt(2.0)))
            return float(np.clip(normalized, 0.0, 1.0))
        except Exception:
            # Fallback si scipy pas dispo
            return normalize_score(score, method="sigmoid")

    else:
        # Pas de normalisation
        return float(score)

def score_with_models(vec: np.ndarray, job_type_small: Optional[str], normalize: bool = True) -> Tuple[Optional[float], Optional[float], float, float, int]:
    if GLOBAL_PACK is None or "cal" not in GLOBAL_PACK or "cutoff" not in GLOBAL_PACK:
        raise HTTPException(status_code=503, detail="GLOBAL_PACK manquant (models/global_pack.pkl)")
    g = float(GLOBAL_PACK["cal"].predict_proba([vec])[0, 1])
    cutoff = float(GLOBAL_PACK["cutoff"])
    s_t = None
    s_final = g
    if job_type_small and job_type_small in TYPE_PACKS:
        pack = TYPE_PACKS[job_type_small]
        if pack and "cal" in pack and "cutoff" in pack:
            s_t = float(pack["cal"].predict_proba([vec])[0, 1])
            a = float(ALPHA_BY_TYPE.get(job_type_small, 0.3))
            s_final = a * g + (1.0 - a) * s_t
            cutoff = float(pack["cutoff"])

    # Appliquer la normalisation si demandée
    if normalize:
        s_final = normalize_score(s_final, method="sigmoid")
        # Ajuster le cutoff également (0.5 après normalisation sigmoïde)
        cutoff = 0.5

    pred = int(s_final >= cutoff)
    return g, s_t, s_final, cutoff, pred

# =========================
# Endpoints
# =========================
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/test")
def load_models_once():
    global GLOBAL_PACK, TYPE_PACKS, TFIDF, EMB_MODEL_NAME
    print("🚀 Chargement des modèles au démarrage...")
    _ = build_features({"job_title": "test", "job_description_text": "dummy"}, {"text_cv_full": "dummy"})
    print("✅ Modèles et encodeurs prêts")


@app.get("/home")
def home():
    """
    Endpoint de bienvenue - Vérifie que tout fonctionne correctement.
    """
    return {
        "message": "✅ API CV Compatibility est en ligne!",
        "version": "2.1",
        "status": "OPERATIONAL",
        "features_count": len(CAND_FEATS) if isinstance(CAND_FEATS, list) else 0,
        "models_loaded": {
            "global": GLOBAL_PACK is not None,
            "type_specific": len(TYPE_PACKS) > 0 if TYPE_PACKS else False,
            "features_order": CAND_FEATS is not None,
            "job_stats": JOB_STATS is not None and len(JOB_STATS.get("job_selectivity_historical", {})) > 0,
            "tfidf": TFIDF is not None,
        },
        "endpoints": {
            "/healthz": "Health check",
            "/home": "Status page (this endpoint)",
            "/model_info": "Detailed model information",
            "/score": "Score a single job-cv pair (POST)",
            "/score_many": "Score multiple pairs (POST)",
            "/score_job/{job_id}": "Score all candidates for a job (GET)",
        },
        "docs": "http://localhost:8000/docs",
    }

@app.get("/model_info")
def model_info():
    return {
        "model_dir": MODEL_DIR,
        "global_loaded": GLOBAL_PACK is not None,
        "type_packs_loaded": len(TYPE_PACKS or {}) if TYPE_PACKS else 0,
        "features_loaded_count": len(CAND_FEATS) if isinstance(CAND_FEATS, list) else None,
        "first_features": CAND_FEATS[:10] if isinstance(CAND_FEATS, list) else None,
        "alpha_by_type": list(ALPHA_BY_TYPE.keys()) if ALPHA_BY_TYPE else [],
        "tfidf_loaded": TFIDF is not None,
        "job_type_clf": bool(JOB_TYPE_CLF is not None and JOB_TYPE_LE is not None),
        "embedder_name": EMB_MODEL_NAME,
    }

@app.post("/score", response_model=ScoreOut)
def score_one(
    payload: OnePair,
    include_features: bool = Query(False, description="Inclure les features dans la réponse"),
    override_type: Optional[str] = Query(None, description="Force le type d'offre (route vers pack spécifique)"),
    normalize: bool = Query(True, description="Normaliser le score pour mieux distribuer (défaut: True)"),
):
    # Prépare job/cv
    if payload.job and payload.cv:
        job = payload.job.dict()
        cv = payload.cv.dict()
    elif payload.job_text and payload.cv_text:
        job = {"job_title": "", "job_description_text": payload.job_text}
        cv = {"text_cv_full": payload.cv_text}
    else:
        raise HTTPException(status_code=400, detail="Fournir {job, cv} ou {job_text, cv_text}")

    feats = build_features(job, cv)
    vec = align_feature_vector(feats, CAND_FEATS if isinstance(CAND_FEATS, list) else None)
    job_type_small = get_job_type(job, feats, override=override_type)
    g, s_t, s_f, cutoff, pred = score_with_models(vec, job_type_small, normalize=normalize)

    out = ScoreOut(
        pair_id=payload.pair_id,
        score_global=g,
        score_type=s_t,
        score_final=s_f,
        cutoff=cutoff,
        pred=pred,
        job_type_small=job_type_small
    )
    if include_features:
        out.features_used = {k: float(v) for k, v in feats.items() if not k.startswith("_")}
    return out

@app.post("/score_many")
def score_many(payload: ManyPairs, include_features: bool = Query(False), override_type: Optional[str] = Query(None), normalize: bool = Query(True)):
    results: List[Dict[str, Any]] = []
    for item in payload.items:
        try:
            # Recompute here to éviter d'appeler l'endpoint /score
            if item.job and item.cv:
                job = item.job.dict()
                cv = item.cv.dict()
            elif item.job_text and item.cv_text:
                job = {"job_title": "", "job_description_text": item.job_text}
                cv = {"text_cv_full": item.cv_text}
            else:
                raise ValueError("Fournir {job, cv} ou {job_text, cv_text}")

            feats = build_features(job, cv)
            vec = align_feature_vector(feats, CAND_FEATS if isinstance(CAND_FEATS, list) else None)
            job_type_small = get_job_type(job, feats, override=override_type)
            g, s_t, s_f, cutoff, pred = score_with_models(vec, job_type_small, normalize=normalize)

            row = {
                "pair_id": item.pair_id,
                "score_global": g,
                "score_type": s_t,
                "score_final": s_f,
                "cutoff": cutoff,
                "pred": pred,
                "job_type_small": job_type_small
            }
            if include_features:
                row["features_used"] = {k: float(v) for k, v in feats.items() if not k.startswith("_")}
            results.append(row)
        except Exception:
            results.append({
                "pair_id": item.pair_id,
                "score_global": None,
                "score_type": None,
                "score_final": float("nan"),
                "cutoff": float("nan"),
                "pred": 0,
                "job_type_small": None,
                "features_used": None
            })
    return {"items": results}

@app.get("/score_job/{job_id}", response_model=ScoreJobOut)
def score_job(
    job_id: str,
    data_path: Optional[str] = Query(None, description="Chemin d'un JSONL GOLD"),
    include_features: bool = Query(False),
    override_type: Optional[str] = Query(None, description="Force le type d'offre"),
    normalize: bool = Query(True, description="Normaliser les scores pour mieux distribuer"),
):
    if pd is None:
        raise HTTPException(status_code=500, detail="pandas non disponible pour lire le JSONL")

    path = data_path or DATA_PATH_DEFAULT
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Fichier introuvable: {path}")

    try:
        df = pd.read_json(path, lines=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lecture JSONL échouée: {e}")

    # Colonne job_id
    col_job_id = None
    for c in ["job_id", "jobId", "jobid"]:
        if c in df.columns:
            col_job_id = c
            break
    if col_job_id is None:
        raise HTTPException(status_code=400, detail="Colonne job_id absente du JSONL")

    sub = df[df[col_job_id].astype(str) == str(job_id)].copy()
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Aucune candidature pour job_id={job_id}")

    # Construire le "job" depuis la première ligne
    row0 = sub.iloc[0].to_dict()
    job = {
        "job_id": job_id,
        "job_title": row0.get("job_title") or row0.get("title") or "",
        "job_description_text": row0.get("job_description_text") or row0.get("description") or "",
        "job_llm_segments": row0.get("job_llm_segments") or {},
        "job_languages_required": row0.get("job_languages_required") or [],
        "job_type_small": row0.get("job_type_small") or None,
        "job_required_years_num": row0.get("job_required_years_num") or None,
        "job_required_skills_plus": row0.get("job_required_skills_plus") or [],
    }
    # Détecter / forcer le type
    job_type_small = get_job_type(job, {}, override=override_type)
    job["job_type_small"] = job_type_small

    # PASS 1: features sans rank -> score provisoire
    tmp = []
    cand_col = None
    for c in ["candidate_id", "candidateId", "candidateID", "cv_candidate_id", "application_candidate_id"]:
        if c in sub.columns:
            cand_col = c
            break
    if cand_col is None:
        if "id" in sub.columns:
            cand_col = "id"
        else:
            raise HTTPException(status_code=400, detail="Colonne candidate_id absente")

    for _, r in sub.iterrows():
        cv = {
            "candidate_id": str(r.get(cand_col)),
            "text_cv_full": r.get("text_cv_full") or "",
            "text_cv_experience": r.get("text_cv_experience") or "",
            "text_cv_skills": r.get("text_cv_skills") or "",
            "cv_languages": r.get("cv_languages") or [],
            "cv_years_experience_num": r.get("cv_years_experience_num") or None,
        }
        feats = build_features(job, cv)
        vec = align_feature_vector(feats, CAND_FEATS if isinstance(CAND_FEATS, list) else None)
        g, s_t, s_f, cutoff, pred = score_with_models(vec, job_type_small, normalize=normalize)
        tmp.append({"cv": cv, "feats": feats, "s_prov": s_f, "cutoff": cutoff, "g": g, "s_t": s_t})

    # Calcul du rang normalisé sur s_prov
    items: List[ScoreJobOutItem] = []
    if tmp:
        scores = np.array([t["s_prov"] for t in tmp], dtype=float)
        order = np.argsort(-scores)  # desc
        n = len(tmp)
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)  # 1..n
        norm = 1.0 - (ranks - 1) / (n - 1) if n > 1 else np.array([1.0])

        # PASS 2: ré-injection du rank -> re-score final
        for i, t in enumerate(tmp):
            f2 = dict(t["feats"])
            f2["rank_in_offer_pool_norm"] = float(norm[i])
            vec2 = align_feature_vector(f2, CAND_FEATS if isinstance(CAND_FEATS, list) else None)
            g2, s_t2, s_f2, cutoff2, pred2 = score_with_models(vec2, job_type_small, normalize=normalize)
            it = ScoreJobOutItem(
                candidate_id=t["cv"]["candidate_id"],
                score_global=g2,
                score_type=s_t2,
                score_final=s_f2,
                cutoff=cutoff2,
                pred=pred2,
                rank_in_offer_pool_norm=float(norm[i]),
            )
            if include_features:
                it.features_used = {k: float(v) for k, v in f2.items() if not k.startswith("_")}
            items.append(it)

    return ScoreJobOut(job_id=str(job_id), job_type_small=job_type_small, n_candidates=len(items), items=items)

# =========================
# Main run (local)
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
