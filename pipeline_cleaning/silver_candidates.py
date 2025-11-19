import os, sys, json, datetime, re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContainerClient, ContentSettings
from azure.core.exceptions import ResourceNotFoundError




AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=absaifabrik;AccountKey=dHnt6TqjK6GYHvEjLTKenFdRHitSCByxcqenpCAvP/+GkY6XjHk7+BMfSpVuhbSpUi/5EfGq61CR+AStw0NiCA==;EndpointSuffix=core.windows.net"
AZURE_BLOB_CONTAINER="cvcompat"
# lecture depuis blob
SOURCE_PREFIX="tt/enriched/candidates/"   
BRONZE_PREFIX="bronze/candidates/"


# Utils
# ==========================
def utc_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)

def safe_get(d: Dict, path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict): return default
        cur = cur.get(p)
        if cur is None: return default
    return cur

def to_iso(ts: Optional[str]) -> Optional[str]:
    if not ts: return None
    # Teamtailor renvoie souvent ISO déjà. On laisse tel quel si c'est déjà ISO.
    return ts

def parse_datetime(ts: Optional[str]) -> Optional[datetime.datetime]:
    if not ts: return None
    try:
        # accepte '...+02:00' etc.
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


# Azure helpers
# ==========================
def get_container() -> ContainerClient:
    container_name = AZURE_BLOB_CONTAINER
    conn_str = AZURE_STORAGE_CONNECTION_STRING
    if not conn_str:
        print("❌ AZURE_STORAGE_CONNECTION_STRING manquante.", file=sys.stderr); sys.exit(1)
    bsc = BlobServiceClient.from_connection_string(conn_str)
    cc = bsc.get_container_client(container_name)
    try:
        cc.get_container_properties()
    except ResourceNotFoundError:
        cc.create_container()
    return cc

def list_json_blobs(container: ContainerClient, prefix: str) -> List[str]:
    return [b.name for b in container.list_blobs(name_starts_with=prefix) if b.name.lower().endswith(".json")]

def download_json(container: ContainerClient, path: str) -> Dict[str, Any]:
    return json.loads(container.get_blob_client(path).download_blob().readall())

def upload_text(container: ContainerClient, path: str, text: str, overwrite: bool=False, content_type: str="application/json; charset=utf-8", metadata: Optional[Dict[str,str]]=None):
    container.get_blob_client(path).upload_blob(
        text.encode("utf-8"), overwrite=overwrite,
        content_settings=ContentSettings(content_type=content_type),
        metadata=metadata
    )

def blob_exists(container: ContainerClient, path: str) -> bool:
    try:
        container.get_blob_client(path).get_blob_properties()
        return True
    except ResourceNotFoundError:
        return False
    


# Extractors from Teamtailor Bronze
# ==========================
def extract_core_from_tt(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = doc.get("data") or {}
    if data.get("type") != "candidates":
        return None
    a = data.get("attributes") or {}
    core = {
        "candidate_id": data.get("id"),
        "first_name": a.get("first-name"),
        "last_name": a.get("last-name"),
        "email": a.get("email"),
        "phone": a.get("phone"),
        "location": safe_get(a, ["location", "city"]) or a.get("location"),
        "linkedin_url": a.get("linkedin-url"),
        "picture_url": safe_get(a, ["picture", "url"]),
        "created_at": to_iso(a.get("created-at")),
        "updated_at": to_iso(a.get("updated-at")),
        "source_site": a.get("referring-site"),
        "referring_url": a.get("referring-url"),
        "internal": a.get("internal"),
        "sourced": a.get("sourced"),
        "unsubscribed": a.get("unsubscribed"),
        "consent_future_jobs_at": to_iso(a.get("consent-future-jobs-at")),
        "original_resume_url": a.get("original-resume"),
    }
    return core

def extract_inc_from_tt(doc: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    included = doc.get("included") or []
    out = {"applications": [], "activities": [], "answers": [], "questions": {}}
    for inc in included:
        t = inc.get("type")
        a = inc.get("attributes") or {}
        if t == "job-applications":
            out["applications"].append({
                "application_id": inc.get("id"),
                "candidate_id": safe_get(inc, ["relationships","candidate","data","id"]),
                "job_id": safe_get(inc, ["relationships","job","data","id"]),
                "status": a.get("status"),
                "created_at": to_iso(a.get("created-at")),
                "updated_at": to_iso(a.get("updated-at")),
                "changed_stage_at": to_iso(a.get("changed-stage-at")),
                "match": a.get("match"),
                "sourced": a.get("sourced"),
                "referring_site": a.get("referring-site"),
                "cover_letter": a.get("cover-letter")
            })
        elif t == "activities":
            out["activities"].append({
                "activity_id": inc.get("id"),
                "created_at": to_iso(a.get("created-at")),
                "code": a.get("code"),              # ex: "stage_changed","email_sent","note_added"...
                "payload": a.get("payload"),        # JSON libre Teamtailor
                "actor_id": safe_get(inc, ["relationships","user","data","id"])
            })
        elif t == "answers":
            out["answers"].append({
                "answer_id": inc.get("id"),
                "candidate_id": safe_get(inc, ["relationships","candidate","data","id"]),
                "question_id": safe_get(inc, ["relationships","question","data","id"]),
                "text": a.get("text"),
                "created_at": to_iso(a.get("created-at"))
            })
        elif t == "questions":
            out["questions"][inc.get("id")] = {
                "title": a.get("title"),
                "answer_type": a.get("answer-type")
            }
    # enrich answers
    if out["answers"] and out["questions"]:
        for x in out["answers"]:
            q = out["questions"].get(x.get("question_id") or "")
            if q:
                x["question_title"] = q["title"]
                x["question_type"] = q["answer_type"]
    return out


# Derivations from activities
# ==========================
STAGE_REGEX = re.compile(r"(stage|pipeline).*", re.I)

def derive_activity_features(activities: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not activities:
        return {
            "last_activity_at": None,
            "interactions_count": 0,
            "counts_by_type": {},
            "stage_transitions": [],
            "time_to_shortlist_hours": None,
            "time_to_interview_hours": None,
            "has_explicit_feedback": False,
            "last_feedback_at": None,
            "labels": []
        }

    # tri par date
    def sort_key(a): 
        dt = parse_datetime(a.get("created_at"))
        return dt or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    acts = sorted(activities, key=sort_key)

    counts: Dict[str,int] = {}
    stage_transitions = []
    feedback_labels = []
    last_feedback_at = None

    shortlist_ts = None
    interview_ts = None
    first_apply_ts = None

    for a in acts:
        code = (a.get("code") or "").lower()
        counts[code] = counts.get(code, 0) + 1
        ts = parse_datetime(a.get("created_at"))

        payload = a.get("payload") or {}
        actor = a.get("actor_id")

        # Stage changes (payload structure varie selon TT ; on essaie plusieurs clés communes)
        if "stage" in code or "pipeline" in code or (isinstance(payload, dict) and ("from" in payload or "to" in payload)):
            from_stage = (payload.get("from") or payload.get("from_stage") or payload.get("previous") or "").lower() if isinstance(payload, dict) else None
            to_stage   = (payload.get("to")   or payload.get("to_stage")   or payload.get("next")     or "").lower() if isinstance(payload, dict) else None
            stage_transitions.append({
                "from_stage": from_stage or None,
                "to_stage": to_stage or None,
                "timestamp": a.get("created_at"),
                "actor": f"user:{actor}" if actor else None
            })
            # heuristiques time-to-*
            name = f"{from_stage}->{to_stage}".lower()
            if not first_apply_ts and (("applied" in from_stage) or ("application" in from_stage)):
                first_apply_ts = ts
            if not shortlist_ts and (("shortlist" in to_stage) or ("screen" in to_stage) or ("review" in to_stage)):
                shortlist_ts = ts
            if not interview_ts and ("interview" in (to_stage or "")):
                interview_ts = ts

        # Feedback explicite : on detecte notes/labels/reject/validate dans code ou payload
        text_dump = ""
        if isinstance(payload, dict):
            text_dump = jdump(payload).lower()
        if any(k in (code + " " + text_dump) for k in ["feedback","validated","rejected","reject","approve","rating","score"]):
            feedback_labels.append("validated" if "valid" in (code+text_dump) else ("rejected" if "reject" in (code+text_dump) else "feedback"))
            last_feedback_at = a.get("created_at")

        # “applied” event fallback
        if not first_apply_ts and ("appl" in code):
            first_apply_ts = ts

    last_activity_at = acts[-1].get("created_at")
    interactions_count = len(acts)

    def hours_between(t0, t1):
        if not (t0 and t1): return None
        return round((t1 - t0).total_seconds() / 3600.0, 2)

    time_to_shortlist = hours_between(first_apply_ts, shortlist_ts)
    time_to_interview = hours_between(first_apply_ts, interview_ts)

    return {
        "last_activity_at": last_activity_at,
        "interactions_count": interactions_count,
        "counts_by_type": counts,
        "stage_transitions": stage_transitions,
        "time_to_shortlist_hours": time_to_shortlist,
        "time_to_interview_hours": time_to_interview,
        "has_explicit_feedback": bool(feedback_labels),
        "last_feedback_at": last_feedback_at,
        "labels": sorted(set(feedback_labels))
    }

# CV segmentation (optional)
# ==========================
def load_cv_segmentation(container: ContainerClient, seg_prefix: str, candidate_id: str) -> Dict[str, Any]:
    path = f"{seg_prefix}{candidate_id}.json"
    if not blob_exists(container, path):
        return {"has_segmented_cv": False}
    try:
        cv = download_json(container, path)
        parsed = cv.get("parsed") or cv  # supporte 2 variantes

        # Extraire les scores de qualité
        quality_scores = parsed.get("quality_scores") or {}

        return {
            "has_segmented_cv": True,
            "profile": {
                "full_name": parsed.get("full_name"),
                "title_or_role": parsed.get("title_or_role"),
                "summary": parsed.get("profile_summary"),
                "location": safe_get(parsed, ["contacts","location"])
            },
            "experiences": parsed.get("experiences") or parsed.get("experience") or [],
            "education": parsed.get("education") or [],
            "skills": parsed.get("skills") or parsed.get("tools_and_technologies") or [],
            "languages": parsed.get("languages") or [],
            "quality_scores": {
                "spelling_score": quality_scores.get("spelling_score"),
                "writing_quality_score": quality_scores.get("writing_quality_score")
            }
        }
    except Exception as e:
        # si lecture impossible, on fallback au flag
        return {"has_segmented_cv": False, "error": str(e)}


# Main: build unified silver
# ==========================
def run():
    load_dotenv()
    container = get_container()

    bronze_candidates = os.getenv("BRONZE_CANDIDATES_PREFIX", "bronze/candidates/")
    seg_prefix = os.getenv("SEGMENTATION_PREFIX", "processed/segmentation/json/")
    silver_unified_prefix = os.getenv("SILVER_UNIFIED_PREFIX", "silver/candidates_unified/")

    run_ts = utc_iso().replace(":", "-")
    out_path = f"{silver_unified_prefix}{run_ts}.jsonl"
    manifest_path = f"silver/_manifests/{run_ts}.json"

    src_paths = list_json_blobs(container, bronze_candidates)
    print(f"🔎 {len(src_paths)} candidats Bronze trouvés sous '{bronze_candidates}'")

    lines = []
    ok, bad = 0, 0

    for src in src_paths:
        try:
            doc = download_json(container, src)
            core = extract_core_from_tt(doc)
            if not core:
                # ignore les non-candidates (par ex. si un fichier job se glisse)
                continue

            inc = extract_inc_from_tt(doc)
            apps = inc["applications"]
            acts = inc["activities"]

            # dérivations d’activité
            act_feat = derive_activity_features(acts)

            # segmentation CV
            cv = load_cv_segmentation(container, seg_prefix, core["candidate_id"])

            # record unifié
            record = {
                **core,
                "applications": apps or [],
                "activity_summary": {
                    "last_activity_at": act_feat["last_activity_at"],
                    "interactions_count": act_feat["interactions_count"],
                    "counts_by_type": act_feat["counts_by_type"],
                    "stage_transitions": act_feat["stage_transitions"],
                    "time_to_shortlist_hours": act_feat["time_to_shortlist_hours"],
                    "time_to_interview_hours": act_feat["time_to_interview_hours"]
                },
                "feedback_signals": {
                    "has_explicit_feedback": act_feat["has_explicit_feedback"],
                    "last_feedback_at": act_feat["last_feedback_at"],
                    "labels": act_feat["labels"]
                },
                "cv": cv
            }
            lines.append(jdump(record))
            ok += 1

        except Exception as e:
            print(f"✗ Erreur lecture/traitement {src}: {e}", file=sys.stderr)
            bad += 1

    if lines:
        upload_text(container, out_path, "\n".join(lines) + "\n", overwrite=False)
    manifest = {
        "run_ts": run_ts,
        "bronze_prefix": bronze_candidates,
        "seg_prefix": seg_prefix,
        "silver_unified_path": out_path,
        "counts": {"ok": ok, "failed": bad}
    }
    upload_text(container, manifest_path, jdump(manifest), overwrite=False)

    print("\n=== FIN ===")
    print(jdump(manifest))

if __name__ == "__main__":
    run()