#!/usr/bin/env python3
"""Redrob candidate ranker — the single reproduce command.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Spec-compliant ranking step: CPU-only, no network, no LLM call, designed to run
the full 100K pool in well under 5 minutes / 16 GB. It loads precomputed
embeddings (artifacts/embeddings.npy) if present for the semantic-recall
component; if absent, semantic falls back to 0 and the rule/career/signal layers
still produce a valid ranking (so the repo is runnable before embeddings exist).

Pipeline:  parse JSONL → extract features → semantic cosine (precomputed)
        → pool-relative recency → score (base × availability × penalties)
        → sort (score desc, then candidate_id asc) → top 100 → grounded reasoning
        → write CSV → self-validate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ranker.features import extract                       # noqa: E402
from ranker.reasoning import reason_for                   # noqa: E402
from ranker.score import score_candidate                  # noqa: E402

ARTIFACTS = HERE / "artifacts"
TOP_N = 100


def load_embeddings():
    """Return (id->cosine-with-JD dict) or None if artifacts are missing.

    IDs are stored as a JSON sidecar (not a pickled object array) so loading is
    safe and dependency-light.
    """
    emb_p = ARTIFACTS / "embeddings.npy"
    ids_p = ARTIFACTS / "emb_ids.json"
    jd_p = ARTIFACTS / "jd_embedding.npy"
    if not (emb_p.exists() and ids_p.exists() and jd_p.exists()):
        return None
    emb = np.load(emb_p).astype(np.float32)       # (N, d), rows L2-normalised
    ids = json.loads(ids_p.read_text(encoding="utf-8"))
    jd = np.load(jd_p).astype(np.float32)         # (d,), normalised
    sims = emb @ jd                               # cosine since both normalised
    lo, hi = float(sims.min()), float(sims.max())
    span = (hi - lo) or 1.0
    sims01 = (sims - lo) / span                    # 0..1 across the pool
    sem = {cid: float(v) for cid, v in zip(ids, sims01)}
    # Neutral default for any candidate missing from the embedding set: the pool
    # median, so a missing vector is treated as average — never as "worst".
    default = float(np.median(sims01)) if len(sims01) else 0.0
    return sem, default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--explain", action="store_true",
                    help="also write submission.debug.jsonl with component breakdowns")
    args = ap.parse_args()

    t0 = time.time()
    loaded = load_embeddings()
    if loaded is None:
        sem, sem_default = None, 0.0
        print("[rank] semantic embeddings: ABSENT (semantic=0 fallback)")
    else:
        sem, sem_default = loaded
        print(f"[rank] semantic embeddings: loaded ({len(sem)} vectors, "
              f"neutral default={sem_default:.3f})")

    # Pass 1: extract features for every candidate (streaming, low memory).
    feats = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            feats.append(extract(json.loads(line)))
    n = len(feats)
    print(f"[rank] extracted features for {n} candidates in {time.time()-t0:.1f}s")

    # Pool-relative recency percentile (1 = most recently active).
    gaps = np.array([fe.last_active_gap_days for fe in feats], dtype=np.float64)
    order = gaps.argsort()                      # ascending gap = more recent first
    recency_pct = np.empty(n)
    recency_pct[order] = 1.0 - np.linspace(0, 1, n, endpoint=False)
    recency_by_id = {feats[i].candidate_id: recency_pct[i] for i in range(n)}

    # Pass 2: score.
    scored = []
    for fe in feats:
        s = sem.get(fe.candidate_id, sem_default) if sem else 0.0
        scored.append(score_candidate(fe, semantic=s,
                                      recency_pct=recency_by_id[fe.candidate_id]))

    # Sort: score desc, then candidate_id asc (the spec's exact tie-break).
    scored.sort(key=lambda x: (-x.score, x.candidate_id))
    top = scored[:TOP_N]

    rows = []
    for rank, s in enumerate(top, start=1):
        rows.append({
            "candidate_id": s.candidate_id,
            "rank": rank,
            "score": f"{s.score:.6f}",
            "reasoning": reason_for(s, rank),
        })

    out_p = Path(args.out)
    with out_p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        w.writerows(rows)
    print(f"[rank] wrote {len(rows)} rows -> {out_p} in {time.time()-t0:.1f}s total")

    if args.explain:
        dbg = out_p.with_suffix(".debug.jsonl")
        with dbg.open("w", encoding="utf-8") as f:
            for rank, s in enumerate(top, start=1):
                f.write(json.dumps({
                    "rank": rank, "candidate_id": s.candidate_id, "score": s.score,
                    "base": s.base, "modifier": s.modifier, "penalty": s.penalty,
                    "components": s.components,
                    "title": s.features.current_title,
                    "honeypot": s.features.honeypot,
                    "stuffer": s.features.keyword_stuffer,
                }) + "\n")
        print(f"[rank] wrote debug breakdown -> {dbg}")

    _self_validate(out_p)


def _self_validate(out_p: Path) -> None:
    try:
        from validate_submission import validate_submission
    except Exception as exc:                                # noqa: BLE001
        print(f"[rank] (validator unavailable: {exc})")
        return
    errors = validate_submission(str(out_p))
    if errors:
        print(f"[rank] !! SUBMISSION INVALID ({len(errors)} issue(s)):")
        for e in errors:
            print("   -", e)
        sys.exit(2)
    print("[rank] submission is valid")


if __name__ == "__main__":
    main()
