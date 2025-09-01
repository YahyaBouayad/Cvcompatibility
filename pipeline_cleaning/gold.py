#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_gold_from_silver.py
- Lit les fichiers SILVER (jobs, candidates, applications) depuis Azure Blob Storage
- Filtre les applications dont le candidat a un CV segmenté (via cv_parsed_full.jsonl OU flag cv.has_segmented_cv)
- Fait la jointure sur job-application -> produit un fichier GOLD JSONL (et optionnellement CSV)
- Respecte la "structure" des SILVER : JSONL, snake_case, champs top-level,
  et conserve les objets imbriqués utiles tels que timings (application) et cv (candidat).
- N'écrit rien en local.

ENV nécessaires:
  AZURE_BLOB_CONNECTION_STRING   (ex: "DefaultEndpointsProtocol=...;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net")
  AZURE_BLOB_CONTAINER           (ex: "cvcompat")

Optionnels (override des préfixes et emplacements):
  SILVER_JOBS_PREFIX             (ex: "silver/jobs/")
  SILVER_CANDIDATES_PREFIX       (ex: "silver/candidates/")
  SILVER_APPLICATIONS_PREFIX     (ex: "silver/applications/")
  CV_PARSED_FULL_PATH            (ex: "processed/segmentation/cv_parsed_full.jsonl")
  GOLD_OUTPUT_PREFIX             (ex: "processed/gold/")

Par défaut, le script sélectionne le fichier le plus récent (Last-Modified) dans chaque préfixe.
"""

import os
import io
import json
import csv
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Iterable

import pandas as pd
from azure.storage.blob import BlobServiceClient

# ========= Config via ENV =========
AZURE_CONN_STR = "DefaultEndpointsProtocol=https;AccountName=absaifabrik;AccountKey=dHnt6TqjK6GYHvEjLTKenFdRHitSCByxcqenpCAvP/+GkY6XjHk7+BMfSpVuhbSpUi/5EfGq61CR+AStw0NiCA==;EndpointSuffix=core.windows.net"
CONTAINER = "cvcompat"

SILVER_JOBS_PREFIX = os.environ.get("SILVER_JOBS_PREFIX", "silver/jobs/")
SILVER_CANDIDATES_PREFIX = os.environ.get("SILVER_CANDIDATES_PREFIX", "silver/candidates_unified/")
SILVER_APPLICATIONS_PREFIX = os.environ.get("SILVER_APPLICATIONS_PREFIX", "silver/job-applications/")

CV_PARSED_FULL_PATH = os.environ.get("CV_PARSED_FULL_PATH", "processed/segmentation/cv_parsed_full.jsonl")
GOLD_OUTPUT_PREFIX = os.environ.get("GOLD_OUTPUT_PREFIX", "gold/")

# Génère un nom horodaté cohérent avec tes fichiers
TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
GOLD_JSONL_BLOB   = f"{GOLD_OUTPUT_PREFIX}gold_applications_{TS}.jsonl"
GOLD_CSV_BLOB     = f"{GOLD_OUTPUT_PREFIX}gold_applications_{TS}.csv"
QUARANTINE_PREFIX = f"{GOLD_OUTPUT_PREFIX}_quarantine/{TS}/"

# ==== Azure helpers ====
def _container():
    if not AZURE_CONN_STR:
        raise RuntimeError("AZURE_BLOB_CONNECTION_STRING manquant.")
    return BlobServiceClient.from_connection_string(AZURE_CONN_STR).get_container_client(CONTAINER)

def list_latest_blob_name(prefix: str) -> Optional[str]:
    cc = _container()
    latest, latest_dt = None, None
    for b in cc.list_blobs(name_starts_with=prefix):
        if not b.size:
            continue
        lm = b.last_modified
        if latest_dt is None or lm > latest_dt:
            latest, latest_dt = b.name, lm
    return latest

def read_blob_bytes(blob_name: str) -> bytes:
    cc = _container()
    return cc.get_blob_client(blob_name).download_blob(max_concurrency=1).readall()

def write_blob_text(blob_name: str, text: str) -> None:
    cc = _container()
    cc.get_blob_client(blob_name).upload_blob(text.encode("utf-8"), overwrite=True)

def write_blob_csv_from_df(blob_name: str, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    cc = _container()
    cc.get_blob_client(blob_name).upload_blob(buf.getvalue().encode("utf-8"), overwrite=True)

# ==== JSONL robuste ====
def iter_json_records_from_bytes(payload: bytes) -> Iterable[Dict[str, Any]]:
    """
    Parser "tolérant" pour JSONL et JSON concaténés:
      - supprime BOM, \x00
      - si le JSONL est bien formé => splitlines + json.loads par ligne
      - si échec => scanner char-par-char avec équilibrage d'accolades { } et gestion des guillemets,
        pour recomposer les objets même si des retours ligne ont été injectés par erreur.
      - records invalides -> levés au caller (qui pourra les mettre en quarantaine)
    """
    text = payload.decode("utf-8", errors="replace").replace("\ufeff", "").replace("\x00", "")
    lines = text.splitlines()
    # Tentative simple: strict-jsonl
    strict_ok = True
    for ln in lines:
        if not ln.strip():
            continue
        try:
            yield json.loads(ln)
        except Exception:
            strict_ok = False
            break
    if strict_ok:
        return

    # Fallback: scanner équilibrage
    buf, objs, depth = [], [], 0
    in_str, esc = False, False
    started = False

    def flush_obj():
        s = "".join(buf).strip()
        buf.clear()
        if not s:
            return None
        return s

    for ch in text:
        buf.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
                started = True
            elif ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth = max(0, depth - 1)
                if depth == 0 and started:
                    raw = flush_obj()
                    if raw:
                        try:
                            objs.append(json.loads(raw))
                        except Exception:
                            # pousser l'erreur au caller via valueError
                            raise ValueError(f"Chunk JSON invalide (taille={len(raw)})")  # noqa
                    started = False

    # Si des objets ont été reconstitués, on les yield
    if objs:
        for o in objs:
            yield o
        return

    # Dernier recours: tout-ou-rien
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            for it in obj:
                yield it
        else:
            yield obj
    except Exception as e:
        raise ValueError(f"Impossible de parser le contenu blob en JSON/JSONL: {e}")

def read_jsonl_from_blob_to_df(blob_name: str, quarantine_name: Optional[str]=None) -> pd.DataFrame:
    """
    Retourne DataFrame + met en quarantaine les records invalides si besoin.
    """
    payload = read_blob_bytes(blob_name)
    rows: List[Dict[str, Any]] = []
    bad: List[str] = []

    try:
        for rec in iter_json_records_from_bytes(payload):
            rows.append(rec)
    except ValueError as e:
        # On tente une passe "ligne à ligne" pour traquer quelques mauvais records
        text = payload.decode("utf-8", errors="replace").replace("\ufeff", "").replace("\x00", "")
        for ln in text.splitlines():
            if not ln.strip():
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                bad.append(ln)
        if bad and quarantine_name:
            write_blob_text(quarantine_name, "\n".join(bad))
        if not rows:
            # rien de lisible
            raise

    df = pd.DataFrame(rows)
    return df

# ==== Chargement SILVER ====
def load_latest_silver() -> Dict[str, pd.DataFrame]:
    jobs_blob = list_latest_blob_name(SILVER_JOBS_PREFIX)
    cands_blob = list_latest_blob_name(SILVER_CANDIDATES_PREFIX)
    apps_blob = list_latest_blob_name(SILVER_APPLICATIONS_PREFIX)
    if not jobs_blob:  raise RuntimeError(f"Aucun fichier sous {SILVER_JOBS_PREFIX}")
    if not cands_blob: raise RuntimeError(f"Aucun fichier sous {SILVER_CANDIDATES_PREFIX}")
    if not apps_blob:  raise RuntimeError(f"Aucun fichier sous {SILVER_APPLICATIONS_PREFIX}")

    print(f"[INFO] Jobs         : {jobs_blob}")
    print(f"[INFO] Candidates   : {cands_blob}")
    print(f"[INFO] Applications : {apps_blob}")

    jobs_df = read_jsonl_from_blob_to_df(jobs_blob, quarantine_name=f"{QUARANTINE_PREFIX}candidates_bad_records.jsonl")
    candidates_df = read_jsonl_from_blob_to_df(cands_blob, quarantine_name=f"{QUARANTINE_PREFIX}candidates_bad_records.jsonl")
    apps_df = read_jsonl_from_blob_to_df(apps_blob, quarantine_name=f"{QUARANTINE_PREFIX}applications_bad_records.jsonl")

    return {"jobs_df": jobs_df, "candidates_df": candidates_df, "apps_df": apps_df}

# ==== CV segmenté ====
def load_segmented_candidate_ids() -> set:
    segmented = set()
    try:
        parsed_df = read_jsonl_from_blob_to_df(CV_PARSED_FULL_PATH, quarantine_name=f"{QUARANTINE_PREFIX}cv_parsed_full_bad.jsonl")
        if "candidate_id" in parsed_df.columns:
            segmented = set(parsed_df["candidate_id"].dropna().astype(str))
        else:
            for _, r in parsed_df.iterrows():
                cid = r.get("candidate_id") or r.get("id") or (r.get("meta", {}) or {}).get("candidate_id")
                if cid is not None:
                    segmented.add(str(cid))
    except Exception as e:
        print(f"[WARN] cv_parsed_full introuvable/illisible: {e}")
    return segmented

# ==== Construction GOLD ====

# ==== Export JSONL/CSV ====
def df_to_jsonl_lines(df: pd.DataFrame) -> List[str]:
    return [json.dumps(rec, ensure_ascii=False) for rec in df.to_dict(orient="records")]

def normalize_cv_value(cv_val):
    """
    Le champ 'cv' peut être:
      - un dict (cas normal),
      - une chaîne JSON (certains exports),
      - None.
    On normalise en dict ou {}.
    """
    if isinstance(cv_val, dict):
        return cv_val
    if isinstance(cv_val, str):
        s = cv_val.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except Exception:
                return {}
    return {}

def has_cv_content(cv_obj: dict) -> bool:
    """
    Assouplie: on considère 'contenu' si au moins un des champs attendus
    existe et n'est pas vide. On tolère aussi 'profile.summary' non vide
    si experiences/education/skills sont absents (cas CV synthétiques).
    """
    if not isinstance(cv_obj, dict):
        return False

    for k in ("experiences", "education", "skills"):
        v = cv_obj.get(k)
        if isinstance(v, list) and len(v) > 0:
            return True
        if isinstance(v, str) and v.strip():
            return True

    # fallback: certains parsers ne renvoient que le résumé
    prof = cv_obj.get("profile") or {}
    if isinstance(prof, dict) and isinstance(prof.get("summary"), str) and prof["summary"].strip():
        return True

    return False

def build_gold() -> pd.DataFrame:
    # Chargement
    D = load_latest_silver()
    jobs_df, candidates_df, apps_df = D["jobs_df"].copy(), D["candidates_df"].copy(), D["apps_df"].copy()

    diag = {}  # petit rapport
    diag["rows_jobs"] = len(jobs_df)
    diag["rows_candidates"] = len(candidates_df)
    diag["rows_applications"] = len(apps_df)

    # IDs en str
    if "candidate_id" in candidates_df: candidates_df["candidate_id"] = candidates_df["candidate_id"].astype(str)
    if "candidate_id" in apps_df:       apps_df["candidate_id"]      = apps_df["candidate_id"].astype(str)
    if "job_id" in apps_df:             apps_df["job_id"]            = apps_df["job_id"].astype(str)
    if "job_id" in jobs_df:             jobs_df["job_id"]            = jobs_df["job_id"].astype(str)

    # Détection candidats segmentés
    segmented_ids = load_segmented_candidate_ids()
    diag["segmented_ids_from_parsed"] = len(segmented_ids)

    if not segmented_ids:
        # fallback flag embedded
        # si 'cv' est un dict dans le DF, on peut tenter d’en extraire has_segmented_cv
        if "cv" in candidates_df.columns and candidates_df["cv"].notna().any():
            cv_norm = []
            for v in candidates_df["cv"]:
                v = normalize_cv_value(v)
                cv_norm.append({"has_segmented_cv": v.get("has_segmented_cv", None)})
            cv_norm = pd.DataFrame(cv_norm)
            candidates_df["cv.has_segmented_cv"] = cv_norm["has_segmented_cv"]

        if "cv.has_segmented_cv" in candidates_df.columns:
            segmented_ids = set(candidates_df.loc[candidates_df["cv.has_segmented_cv"] == True, "candidate_id"].astype(str))

    diag["segmented_ids_final"] = len(segmented_ids)

    # Filtre applications
    apps_f = apps_df[apps_df["candidate_id"].isin(segmented_ids)].copy()
    diag["rows_apps_after_segmented"] = len(apps_f)

    # Sélection "style silver"
    app_keep = [
        "application_id","candidate_id","job_id",
        "decision","stage_id","stage_name",
        "reject_reason_id","reject_reason_text",
        "created_at","updated_at",
        "timings",
        "job_title","job_status","job_employment_type","job_employment_level",
        "job_language_code","job_remote_status","job_created_at","job_updated_at",
        "source_site","source_url"
    ]
    apps_sel = apps_f[[c for c in app_keep if c in apps_f.columns]].copy()

    cand_keep = [
        "candidate_id","first_name","last_name","email","phone","location",
        "cv"  # on garde le bloc tel quel
    ]
    cands_sel = candidates_df[[c for c in cand_keep if c in candidates_df.columns]].copy()

    jobs_keep = [
        "job_id","title","status","employment_type","locations",
        "body","has_description","body_word_count"
    ]
    jobs_sel = jobs_df[[c for c in jobs_keep if c in jobs_df.columns]].copy()

    # Merge
    gold = apps_sel.merge(cands_sel, how="left", on="candidate_id").merge(jobs_sel, how="left", on="job_id")
    diag["rows_after_merge"] = len(gold)

    # Normaliser le champ cv (si chaîne JSON -> dict)
    if "cv" in gold.columns:
        gold["cv"] = gold["cv"].apply(normalize_cv_value)

    

    # Echantillons pour debug rapide
    def head5(df, cols):
        cols = [c for c in cols if c in df.columns]
        return df[cols].head(5).to_dict(orient="records")

    diag["sample_apps_before"] = head5(apps_df, ["application_id","candidate_id","job_id","decision"])
    diag["sample_apps_after_segmented"] = head5(apps_f, ["application_id","candidate_id","job_id","decision"])
    diag["sample_gold_after_merge"] = head5(gold, ["application_id","candidate_id","job_id","decision"])
    diag["missing_cv_count"] = int((gold["cv"].isna() | (gold["cv"].apply(lambda x: not isinstance(x, dict)))).sum()) if "cv" in gold.columns else None

    # Dépose le rapport diag dans le blob
    try:
        diag_blob = f"{GOLD_OUTPUT_PREFIX}_diagnostics/diag_{TS}.json"
        write_blob_text(diag_blob, json.dumps(diag, ensure_ascii=False, indent=2))
        print(f"[INFO] Rapport diag écrit: {diag_blob}")
    except Exception as e:
        print(f"[WARN] Impossible d'écrire le diag: {e}")

    return gold

def main():
    gold_df = build_gold()
    print(f"[INFO] GOLD rows: {len(gold_df)} | cols: {len(gold_df.columns)}")

    # JSONL
    write_blob_text(GOLD_JSONL_BLOB, "\n".join([json.dumps(rec, ensure_ascii=False) for rec in gold_df.to_dict(orient="records")]) + ("\n" if len(gold_df) else ""))
    print(f"[OK] Écrit: {GOLD_JSONL_BLOB}")

    # CSV (optionnel)
    try:
        write_blob_csv_from_df(GOLD_CSV_BLOB, gold_df)
        print(f"[OK] Écrit: {GOLD_CSV_BLOB}")
    except Exception as e:
        print(f"[WARN] CSV non écrit: {e}")



if __name__ == "__main__":
    main()