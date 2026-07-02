#!/usr/bin/env python3
"""OFFLINE: embed the JD and every candidate's composed text with a small local
sentence-transformer, and save the matrix as a precomputed artifact.

This is the ONLY place a model runs. It is offline (the spec allows pre-computation
that exceeds the 5-minute window); the shipped ``rank.py`` merely loads the saved
``embeddings.npy`` and does NumPy cosine — no torch, no network at rank time.

Output artifacts (committed so Stage-3 reproduction needs no model download):
    artifacts/embeddings.npy   float16  (N, dim)   L2-normalised rows
    artifacts/emb_ids.json     JSON list (N,)       candidate_id order
    artifacts/jd_embedding.npy float16  (dim,)     normalised JD vector

Run:
    pip install -r requirements-offline.txt
    python precompute/embed_candidates.py --candidates "<path>/candidates.jsonl"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384-dim, CPU-friendly
MAX_CHARS = 900     # ~220 tokens; front-loads headline+summary+early career signal
MAX_SEQ_LEN = 256   # cap the encoder sequence length for CPU throughput


def compose_text(c: dict) -> str:
    """The candidate text we embed — biased toward career-history *substance*
    (descriptions) over the skills array, so plain-language Tier-5s surface and
    keyword-stuffer skill lists don't dominate the vector.

    Truncated to ``MAX_CHARS`` so CPU embedding of 100K docs stays tractable; the
    composition front-loads the highest-signal fields (headline, summary, then the
    most recent roles) so truncation drops the least informative tail."""
    p = c.get("profile", {}) or {}
    parts = [p.get("headline", ""), p.get("summary", "")]
    for j in (c.get("career_history", []) or [])[:5]:
        parts.append(f"{j.get('title','')} at {j.get('company','')} ({j.get('industry','')}): "
                     f"{j.get('description','')}")
    # a light skills hint (names only), appended last so it's not the focus
    names = ", ".join(s.get("name", "") for s in (c.get("skills", []) or [])[:15])
    parts.append("Skills: " + names)
    return "\n".join(x for x in parts if x).strip()[:MAX_CHARS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    import os
    os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    from sentence_transformers import SentenceTransformer
    from ranker.jd_rubric import JD_QUERY_TEXT

    ARTIFACTS.mkdir(exist_ok=True)
    model = SentenceTransformer(args.model, device="cpu")
    model.max_seq_length = MAX_SEQ_LEN

    ids: list[str] = []
    texts: list[str] = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            ids.append(c.get("candidate_id", ""))
            texts.append(compose_text(c))

    print(f"Embedding {len(texts)} candidates with {args.model} (CPU)...")
    # bge models recommend a query prefix; candidates are 'passages'.
    emb = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    jd = model.encode([f"Represent this sentence for searching relevant passages: {JD_QUERY_TEXT}"],
                      normalize_embeddings=True, convert_to_numpy=True)[0]

    np.save(ARTIFACTS / "embeddings.npy", emb.astype(np.float16))
    (ARTIFACTS / "emb_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    np.save(ARTIFACTS / "jd_embedding.npy", jd.astype(np.float16))
    print(f"Saved artifacts to {ARTIFACTS} (matrix {emb.shape}, {emb.nbytes//2//1024//1024} MB fp16)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
