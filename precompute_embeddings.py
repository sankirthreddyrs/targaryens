import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import argparse
from datetime import datetime, timezone

def get_args():
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings (offline step — no time limit)")
    parser.add_argument("--candidates", type=str, default="candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--out_pkl", type=str, default="precomputed_data.pkl",
                        help="Path to save precomputed_data.pkl")
    return parser.parse_args()

# ── Honeypot detection ────────────────────────────────────────────────────────
def is_honeypot(cand):
    """
    Detect candidates with structurally impossible profiles.
    The spec says ~80 honeypots exist; ranking them high = disqualification.
    """
    yoe = cand.get("profile", {}).get("years_of_experience", 0)

    for job in cand.get("career_history", []):
        if job.get("duration_months", 0) > (yoe * 12 + 12):
            return True

    for edu in cand.get("education", []):
        if edu.get("start_year", 0) > edu.get("end_year", 9999):
            return True

    for skill in cand.get("skills", []):
        if skill.get("duration_months", 0) > (yoe * 12 + 24):
            return True
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 0) == 0:
            return True

    return False

# ── Recency scoring ───────────────────────────────────────────────────────────
def compute_recency_score(last_active_date_str):
    """
    The JD explicitly says: down-weight candidates who haven't logged in for 6 months.
    Returns 0.0–1.0.
    """
    if not last_active_date_str:
        return 0.25
    try:
        last_active = datetime.strptime(last_active_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        days_ago = (datetime.now(timezone.utc) - last_active).days
        if days_ago <= 30:    return 1.0
        elif days_ago <= 60:  return 0.85
        elif days_ago <= 90:  return 0.65
        elif days_ago <= 180: return 0.35
        else:                 return 0.05   # 6+ months inactive = hard down-weight
    except Exception:
        return 0.25

# ── Title filter ──────────────────────────────────────────────────────────────
# STRATEGY: broad exclusion of roles that are structurally incompatible with AI engineering.
# We deliberately KEEP ambiguous titles (software engineer, backend engineer, data analyst)
# because the JD explicitly says: "a backend engineer who built a recommendation system
# at a product company IS a fit." Semantic similarity handles the discrimination.
#
# We only hard-exclude titles where there is ZERO plausible path to this role:
# BAs, PMs, Java devs, mobile devs, .NET devs, DevOps, frontend, QA, full-stack.
HARD_EXCLUDE_TITLES = {
    "business analyst", "project manager", "java developer", "mobile developer",
    ".net developer", "devops engineer", "frontend engineer", "qa engineer",
    "full stack developer", "cloud engineer",
}

def is_excluded_title(title: str) -> bool:
    t = title.lower()
    return any(excl in t for excl in HARD_EXCLUDE_TITLES)

# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text_for_embedding(cand):
    """
    Build a rich, comprehensive text blob for semantic embedding.
    Includes ALL job history (not just last 2), ALL skills, education.
    The JD hint: a candidate whose CAREER HISTORY shows recommendation system
    experience is a fit even without AI keywords in the title.
    Full history is essential to capture this.
    """
    profile = cand.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]

    for job in cand.get("career_history", []):
        title   = job.get("title", "")
        company = job.get("company", "")
        desc    = job.get("description", "")
        if title:   parts.append(f"{title} at {company}")
        if desc:    parts.append(desc)

    skill_names = [s.get("name", "") for s in cand.get("skills", []) if s.get("name")]
    if skill_names:
        parts.append("Technical skills: " + ", ".join(skill_names))

    for edu in cand.get("education", []):
        degree = edu.get("degree", "")
        field  = edu.get("field_of_study", "")
        if degree or field:
            parts.append(f"{degree} in {field}".strip(" in"))

    return " ".join(filter(None, parts))

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(cand):
    """Extract all structured signals used by rank.py scoring."""
    profile  = cand.get("profile", {})
    signals  = cand.get("redrob_signals", {})
    history  = cand.get("career_history", [])
    skills   = cand.get("skills", [])

    # The JD explicitly names these firms as disqualifiers if entire career is there
    consulting_firms = {
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
        "mindtree", "tech mahindra", "mphasis", "hexaware", "hcl technologies",
        "hcl", "ibm global services", "ltimindtree"
    }
    has_product = any(
        job.get("company", "").lower().strip() not in consulting_firms
        for job in history
    )

    # Skill names for keyword matching in rank.py
    skill_names = [s.get("name", "").lower() for s in skills if s.get("name")]

    last_active = signals.get("last_active_date", "")
    gh          = signals.get("github_activity_score", -1)

    return {
        "candidate_id":            cand["candidate_id"],
        "yoe":                     profile.get("years_of_experience", 0),
        "current_title":           profile.get("current_title", ""),
        "has_product":             has_product,
        "location":                profile.get("location", "").lower(),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0),
        "last_active_date":        last_active,
        "recency_score":           compute_recency_score(last_active),
        "notice_period_days":      signals.get("notice_period_days", 90),
        "github_activity_score":   gh if gh >= 0 else None,
        "open_to_work_flag":       signals.get("open_to_work_flag", False),
        "skill_names":             skill_names,
        "honeypot":                False,  # honeypots are dropped entirely, never stored
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    wall_start = time.time()

    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    candidates_data, texts = [], []
    dropped = {"honeypot": 0, "title": 0, "yoe": 0}

    print(f"Reading candidates from: {args.candidates}")
    with open(args.candidates, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            cand = json.loads(line)

            # Gate 1 — honeypot (structurally impossible profiles)
            if is_honeypot(cand):
                dropped["honeypot"] += 1
                continue

            # Gate 2 — structurally incompatible title
            title = cand.get("profile", {}).get("current_title", "")
            if is_excluded_title(title):
                dropped["title"] += 1
                continue

            # Gate 3 — too junior (< 2 YOE)
            if cand.get("profile", {}).get("years_of_experience", 0) < 2:
                dropped["yoe"] += 1
                continue

            candidates_data.append(extract_features(cand))
            texts.append(extract_text_for_embedding(cand))

            if (idx + 1) % 20000 == 0:
                print(f"  Scanned {idx+1:,} | Kept: {len(texts):,} | "
                      f"Dropped — honeypot:{dropped['honeypot']} "
                      f"title:{dropped['title']} yoe:{dropped['yoe']}")

    t_filter = time.time() - wall_start
    total_dropped = sum(dropped.values())
    print(f"\nFiltering done in {t_filter:.1f}s")
    print(f"Kept for embedding: {len(texts):,} | Total dropped: {total_dropped:,}")
    print(f"  honeypot:{dropped['honeypot']}  bad_title:{dropped['title']}  too_junior:{dropped['yoe']}")

    print("\nGenerating embeddings (this is the slow offline step — no time limit applies)...")
    t_embed = time.time()
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # pre-normalise: dot product = cosine at rank time
    )
    print(f"Embeddings done in {time.time() - t_embed:.1f}s")

    print(f"Saving to {args.out_pkl}...")
    with open(args.out_pkl, "wb") as f:
        pickle.dump({"features": candidates_data, "embeddings": embeddings}, f, protocol=4)

    print(f"\nAll done. Total wall time: {time.time() - wall_start:.1f}s")
    print(f"PKL contains {len(candidates_data):,} candidates ready for rank.py.")

if __name__ == "__main__":
    main()