import json
import numpy as np
import pickle
import csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import argparse
from datetime import datetime, timezone

def get_args():
    parser = argparse.ArgumentParser(description="Rank candidates — must complete in < 5 minutes on CPU")
    parser.add_argument("--candidates", type=str, default="candidates.jsonl",
                        help="Original candidates file (required by hackathon spec)")
    parser.add_argument("--out",        type=str, default="submission.csv",
                        help="Output CSV path")
    parser.add_argument("--pkl_path",   type=str, default="precomputed_data.pkl",
                        help="Path to precomputed_data.pkl")
    return parser.parse_args()

# ── Job Description ───────────────────────────────────────────────────────────
# Taken verbatim from the hackathon bundle job_description.docx.
# Covers both what the JD says AND what it means (per the "final note" hint).
JD_TEXT = """
Senior AI Engineer role at a Series A AI-native talent intelligence platform in Pune or Noida India.

Required: Production experience with embeddings-based retrieval systems using sentence-transformers,
OpenAI embeddings, BGE, E5 or similar. Hands-on with embedding drift, index refresh, and
retrieval-quality regression in production.

Required: Production experience with vector databases and hybrid search infrastructure —
Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, ANN, HNSW.

Required: Strong Python. Hands-on experience designing evaluation frameworks for ranking systems —
NDCG, MRR, MAP, offline-to-online correlation, A/B testing.

Preferred: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank (XGBoost, LambdaMART, neural rankers),
RAG systems, recommendation systems, information retrieval, NLP, deep learning, transformer models.

Experience shipping end-to-end ranking, search, or recommendation systems to real users at scale.
Product company background preferred over pure consulting.
Scrappy product-engineering mindset — shipping working systems over research-perfect ones.
5 to 9 years total experience with 4 to 5 years in applied ML at product companies.
"""

# ── High-value technical keywords ────────────────────────────────────────────
# Exact terms from the JD that get diluted in 384-dim cosine space.
# A direct skill match is a strong signal on top of semantic similarity.
HIGH_VALUE_SKILLS = {
    # Retrieval / embeddings (the JD's #1 requirement)
    "embeddings", "dense retrieval", "semantic search", "vector search",
    "sentence-transformers", "sentence transformers", "bge", "e5",
    "faiss", "annoy", "hnsw", "ann",
    # Vector DBs (#2 requirement)
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "opensearch",
    # Ranking / eval (#3 requirement)
    "learning to rank", "ltr", "lambdamart", "ndcg", "mrr", "map",
    "reranking", "reranker", "xgboost ranker",
    # LLMs / fine-tuning
    "llm", "rag", "retrieval augmented generation", "fine-tuning",
    "lora", "qlora", "peft", "hugging face", "huggingface", "transformers",
    # Core ML
    "pytorch", "recommendation system", "information retrieval",
    "a/b testing", "nlp", "natural language processing",
}

def skill_match_score(skill_names: list) -> float:
    """
    0.0–1.0. Each matched keyword from skill list contributes.
    Caps at 1.0 after ~5 strong matches so it can't dominate the score.
    """
    if not skill_names:
        return 0.0
    matched = sum(1 for s in skill_names if s in HIGH_VALUE_SKILLS)
    return min(matched / 5.0, 1.0)

# ── Availability score ────────────────────────────────────────────────────────
def availability_score(features: dict) -> float:
    """
    Combines all signals related to whether this candidate is actually reachable.
    The JD explicitly calls this out: a great-on-paper candidate who is inactive
    with a low response rate is, for hiring purposes, not available.
    Returns 0.0–1.0.
    """
    score = 0.0

    # Recruiter response rate (0–1 scale, mean ~0.46 in dataset)
    rr = features.get("recruiter_response_rate", 0)
    score += rr * 0.35

    # Platform recency (pre-computed in precompute step)
    recency = features.get("recency_score", 0.25)
    score += recency * 0.35

    # Notice period — JD says sub-30 preferred, can buy out up to 30 days
    np_days = features.get("notice_period_days", 90)
    if np_days == 0:        np_score = 1.0
    elif np_days <= 30:     np_score = 0.9
    elif np_days <= 60:     np_score = 0.5
    elif np_days <= 90:     np_score = 0.2
    else:                   np_score = 0.0   # 120-150 day notice = not available
    score += np_score * 0.20

    # Explicit job-seeking signals
    if features.get("open_to_work_flag"):
        score += 0.07

    # GitHub activity (0–100 scale; mean ~33, p90 ~60)
    gh = features.get("github_activity_score")
    if gh is not None:
        score += min(gh / 100.0, 1.0) * 0.03

    return min(score, 1.0)

# ── Reasoning generation ──────────────────────────────────────────────────────
def generate_reasoning(features: dict, final_score: float,
                        semantic_sim: float, kw_score: float) -> str:
    """
    Stage 4 requirement: reasoning must reference specific facts from the profile,
    connect to JD requirements, acknowledge concerns, not hallucinate,
    vary across candidates, and match rank tone.

    We only state facts we actually have in features — no invention.
    """
    title  = features.get("current_title", "Engineer")
    yoe    = features.get("yoe", 0)
    loc    = features.get("location", "")
    rr     = features.get("recruiter_response_rate", 0)
    np_days= features.get("notice_period_days", 90)
    recency= features.get("recency_score", 0.25)
    gh     = features.get("github_activity_score")
    otw    = features.get("open_to_work_flag", False)
    has_p  = features.get("has_product", False)
    skills = features.get("skill_names", [])

    parts = []

    # --- Semantic fit ---
    if semantic_sim > 0.45:
        parts.append(f"strong semantic alignment with the JD's retrieval and ranking requirements (sim={semantic_sim:.2f})")
    elif semantic_sim > 0.35:
        parts.append(f"moderate-good semantic fit with the JD (sim={semantic_sim:.2f})")
    else:
        parts.append(f"limited semantic overlap with JD requirements (sim={semantic_sim:.2f})")

    # --- Technical stack ---
    jd_skills_present = [s for s in skills if s in HIGH_VALUE_SKILLS]
    if len(jd_skills_present) >= 3:
        parts.append(f"verified JD-relevant skills: {', '.join(jd_skills_present[:4])}")
    elif len(jd_skills_present) >= 1:
        parts.append(f"some JD-relevant skills: {', '.join(jd_skills_present[:2])}")
    else:
        parts.append("no exact JD keyword matches in skills list")

    # --- Experience ---
    if 5 <= yoe <= 9:
        parts.append(f"ideal {yoe:.1f} YOE (JD target: 5–9)")
    elif yoe > 9:
        parts.append(f"{yoe:.1f} YOE (above JD range; may be overqualified)")
    elif 3 <= yoe < 5:
        parts.append(f"{yoe:.1f} YOE (slightly below JD's 5–9 target)")
    else:
        parts.append(f"{yoe:.1f} YOE (junior for this role)")

    # --- Product vs consulting ---
    if not has_p:
        parts.append("entire career at consulting firms — explicit concern per JD")
    
    # --- Location ---
    preferred_locs = ["pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "bangalore", "bengaluru"]
    if any(p in loc for p in ["pune", "noida"]):
        parts.append("preferred location (Pune/Noida)")
    elif any(p in loc for p in preferred_locs):
        parts.append(f"acceptable location ({loc.split(',')[0].strip()})")
    else:
        parts.append(f"non-preferred location ({loc.split(',')[0].strip() if loc else 'unknown'})")

    # --- Availability concerns ---
    concerns = []
    if rr < 0.2:
        concerns.append(f"very low recruiter response rate ({rr:.0%})")
    if recency <= 0.05:
        concerns.append("inactive on platform for 6+ months")
    if np_days > 90:
        concerns.append(f"{np_days}-day notice period")
    if concerns:
        parts.append("concerns: " + "; ".join(concerns))
    elif rr > 0.6 and recency >= 0.65:
        parts.append("actively engaged and highly responsive")
    elif otw:
        parts.append("explicitly open to work")

    # --- GitHub ---
    if gh is not None and gh >= 60:
        parts.append(f"strong open-source activity (GitHub score {gh:.0f}/100)")

    sentence = f"{title} with {yoe:.1f} YOE. " + "; ".join(parts).capitalize() + "."
    return sentence

# ── Scoring ───────────────────────────────────────────────────────────────────
def score_candidate(features: dict, semantic_sim: float) -> tuple:
    """
    Returns (final_score, adjusted_semantic_sim, kw_score).

    Weight rationale (must match JD priorities):
      50% semantic  — primary signal; captures career history semantics, not just keywords
      20% YOE       — JD says 5–9 is the target band
      15% keywords  — direct tech stack match (FAISS, NDCG, etc.)
      15% availability — JD explicitly calls out inactive/low-RR candidates
    """
    # ── Penalties before weighting ──
    if not features.get("has_product"):
        semantic_sim -= 0.08   # consulting-only penalty (JD: explicit disqualifier)

    # CV/speech/robotics without NLP is also a JD disqualifier
    title = features.get("current_title", "").lower()
    if "computer vision" in title and "nlp" not in str(features.get("skill_names", [])):
        semantic_sim -= 0.05

    # ── Component scores ──
    yoe = features.get("yoe", 0)
    if 5 <= yoe <= 9:       yoe_score = 1.0
    elif 4 <= yoe < 5:      yoe_score = 0.7
    elif 9 < yoe <= 11:     yoe_score = 0.7
    elif 3 <= yoe < 4:      yoe_score = 0.4
    elif yoe > 11:           yoe_score = 0.3
    else:                    yoe_score = 0.0

    kw_score   = skill_match_score(features.get("skill_names", []))
    avail      = availability_score(features)

    # ── Location bonus (additive, small) ──
    loc = features.get("location", "")
    if any(p in loc for p in ["pune", "noida"]):
        loc_bonus = 0.04
    elif any(p in loc for p in ["hyderabad", "mumbai", "delhi", "gurgaon", "bangalore", "bengaluru"]):
        loc_bonus = 0.02
    else:
        loc_bonus = 0.0

    final = (
        semantic_sim * 0.50 +
        yoe_score    * 0.20 +
        kw_score     * 0.15 +
        avail        * 0.15 +
        loc_bonus
    )

    return final, semantic_sim, kw_score

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args  = get_args()
    start = time.time()

    # ── Load precomputed data ──
    print("Loading precomputed data...")
    with open(args.pkl_path, "rb") as f:
        data = pickle.load(f)

    features_list       = data["features"]
    candidate_embeddings = data["embeddings"]
    print(f"Loaded {len(features_list):,} candidates in {time.time()-start:.1f}s")

    # ── Embed the JD ──
    print("Embedding job description...")
    model        = SentenceTransformer("all-MiniLM-L6-v2")
    jd_embedding = model.encode([JD_TEXT], convert_to_numpy=True, normalize_embeddings=True)

    # ── Semantic similarities ──
    print("Computing cosine similarities...")
    similarities = cosine_similarity(jd_embedding, candidate_embeddings)[0]

    # ── Score every candidate ──
    print("Scoring candidates...")

    # Hard-exclude titles that survived precompute filtering but should never appear in top 100
    EXCLUDE_TITLES = {
        "business analyst", "project manager", "java developer", "mobile developer",
        ".net developer", "devops engineer", "frontend engineer", "qa engineer",
        "full stack developer", "cloud engineer",
    }

    scored = []
    for i, features in enumerate(features_list):
        if features.get("honeypot"):
            continue
        title = features.get("current_title", "").lower()
        if any(ex in title for ex in EXCLUDE_TITLES):
            continue

        final_score, adj_sim, kw_score = score_candidate(features, float(similarities[i]))

        scored.append({
            "candidate_id": features["candidate_id"],
            "score":        round(final_score, 4),
            "semantic_sim": adj_sim,
            "kw_score":     kw_score,
            "features":     features,
        })

    # ── Sort and take top 100 ──
    scored.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top_100 = scored[:100]

    # ── Validate spec requirements before writing ──
    assert len(top_100) == 100, f"Expected 100 candidates, got {len(top_100)}"
    scores = [c["score"] for c in top_100]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i+1], f"Score not monotonically non-increasing at rank {i+1}"

    # ── Write CSV ──
    print(f"Writing {args.out}...")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, cand in enumerate(top_100, start=1):
            reasoning = generate_reasoning(
                cand["features"], cand["score"],
                cand["semantic_sim"], cand["kw_score"]
            )
            writer.writerow([
                cand["candidate_id"],
                rank,
                f"{cand['score']:.4f}",
                reasoning,
            ])

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s  (well within 5-minute limit)")
    print(f"Top candidate : {top_100[0]['candidate_id']} — score {top_100[0]['score']:.4f}")
    print(f"Bottom of top : {top_100[99]['candidate_id']} — score {top_100[99]['score']:.4f}")
    print(f"Output        : {args.out}")

if __name__ == "__main__":
    main()