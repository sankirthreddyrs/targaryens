# Redrob Intelligent Candidate Ranking

This repository contains our team's AI ranking system for the Redrob India Runs Data and AI Challenge.

## Architecture & Methodology
We implemented a **Two-Phase Hybrid Architecture** to overcome the strict 5-minute CPU constraint while maintaining true semantic understanding:

1. **Phase 1 (Offline Pre-computation - No Time Limit):** 
   - We use a local Sentence Transformer (`all-MiniLM-L6-v2`) to mathematically embed the candidates' ENTIRE career history and skill sets.
   - We deliberately chose NOT to use cheap keyword-filtering here. We accept a ~45 minute offline compute time to ensure maximum accuracy (e.g., catching a backend engineer who secretly built a recommendation system).
   - This generates a heavily pre-processed vector store (`precomputed_data.pkl`) containing everything needed for scoring.

2. **Phase 2 (Fast Ranking - `rank.py` - Strict 5-Minute Limit):**
   - The ranking script embeds the Job Description's core requirements.
   - It performs fast cosine-similarity against the pre-calculated candidate vectors.
   - It layers on an `availability_score` (GitHub, Response Rate, Recency, Notice Period) and direct Keyword boosting.
   - Automatically generates a 1-2 sentence justification explaining the exact reasoning behind the rank.
   - **Performance:** Runs in < 5 seconds on a standard CPU, easily beating the 5-minute limit.

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Reproducing the Output

### Step 1: Pre-compute Embeddings
If testing with a new dataset, you must run the pre-computation script first.
```bash
python precompute_embeddings.py --candidates path/to/candidates.jsonl --out_pkl precomputed_data.pkl
```

### Step 2: Run the Ranker
As required by the hackathon spec, the final ranking command is:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --pkl_path precomputed_data.pkl
```
*(Note: `rank.py` pulls its data from the `.pkl` file, but accepts the `--candidates` argument to comply with the evaluation spec).*

## Section 10.5 Compliance (Docker Sandbox)
Per the hackathon specification (Section 10.5), if a hosted sandbox is impractical, a self-contained Docker recipe must be provided. 

To run the ranking system end-to-end within the ≤5 minute compute budget using Docker:

1. Build the image:
```bash
docker build -t redrob-ranker .
```

2. Run the ranker:
```bash
docker run --rm -v $(pwd):/app redrob-ranker --candidates ./candidates.jsonl --out ./team_submission.csv --pkl_path ./precomputed_data.pkl
```
