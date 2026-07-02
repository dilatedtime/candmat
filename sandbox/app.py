"""candmat sandbox — a small Streamlit UI to run the ranker on a candidate sample.

Satisfies the challenge's sandbox requirement: accept a small candidate sample
(<=100), run the ranking end-to-end on CPU, and show the ranked output with
reasoning. Reuses the exact production ranker modules — no reimplementation, so
what you see here is what rank.py produces.

Run locally:   streamlit run sandbox/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ranker.features import extract                      # noqa: E402
from ranker.reasoning import reason_for                  # noqa: E402
from ranker.score import score_candidate                 # noqa: E402

ARTIFACTS = ROOT / "artifacts"
SAMPLE = ROOT / "data" / "ci_candidates.jsonl"

st.set_page_config(page_title="candmat — Candidate Ranker", page_icon="🎯", layout="wide")


@st.cache_resource
def load_semantic():
    emb_p, ids_p, jd_p = (ARTIFACTS / "embeddings.npy",
                          ARTIFACTS / "emb_ids.json",
                          ARTIFACTS / "jd_embedding.npy")
    if not (emb_p.exists() and ids_p.exists() and jd_p.exists()):
        return None, 0.0
    emb = np.load(emb_p).astype(np.float32)
    ids = json.loads(ids_p.read_text(encoding="utf-8"))
    jd = np.load(jd_p).astype(np.float32)
    sims = emb @ jd
    lo, hi = float(sims.min()), float(sims.max())
    sims01 = (sims - lo) / ((hi - lo) or 1.0)
    return {cid: float(v) for cid, v in zip(ids, sims01)}, float(np.median(sims01))


def rank(candidates: list[dict], top_n: int):
    sem, sem_default = load_semantic()
    feats = [extract(c) for c in candidates]
    gaps = np.array([f.last_active_gap_days for f in feats], dtype=np.float64)
    n = len(feats)
    recency = np.empty(n)
    if n:
        order = gaps.argsort()
        recency[order] = 1.0 - np.linspace(0, 1, n, endpoint=False)
    scored = []
    for i, f in enumerate(feats):
        s = sem.get(f.candidate_id, sem_default) if sem else 0.0
        scored.append(score_candidate(f, semantic=s, recency_pct=recency[i] if n else 0.5))
    scored.sort(key=lambda x: (-x.score, x.candidate_id))
    return scored[:top_n], (sem is not None)


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🎯 candmat — Intelligent Candidate Ranker")
st.caption("Redrob challenge · ranks candidates against the Senior AI Engineer JD by "
           "understanding the role, not matching keywords. CPU-only, no network at rank time.")

with st.sidebar:
    st.header("Input")
    src = st.radio("Candidate source", ["Bundled sample (200)", "Upload JSONL"])
    top_n = st.slider("Show top N", 5, 100, 25)
    uploaded = None
    if src == "Upload JSONL":
        uploaded = st.file_uploader("candidates .jsonl (one JSON object per line)", type=["jsonl", "json"])
    st.markdown("---")
    st.markdown("**How it scores**: semantic (0.30) · trusted skills (0.28) · "
                "career evidence (0.24) · experience (0.10) · context (0.08), "
                "then × availability × trap penalties.")
    go = st.button("Rank candidates", type="primary")


def read_candidates() -> list[dict]:
    if src == "Upload JSONL" and uploaded is not None:
        text = uploaded.getvalue().decode("utf-8")
        if uploaded.name.endswith(".json") and text.lstrip().startswith("["):
            return json.loads(text)
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    if SAMPLE.exists():
        return [json.loads(l) for l in SAMPLE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return []


if go:
    candidates = read_candidates()
    if not candidates:
        st.error("No candidates loaded. Upload a JSONL file or ensure the bundled sample exists.")
        st.stop()
    if len(candidates) > 100:
        st.warning(f"Sample has {len(candidates)} candidates; the sandbox caps at 100 per the spec. Using the first 100.")
        candidates = candidates[:100]

    with st.spinner(f"Ranking {len(candidates)} candidates on CPU…"):
        top, sem_on = rank(candidates, top_n)

    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates ranked", len(candidates))
    c2.metric("Semantic layer", "ON" if sem_on else "fallback")
    c3.metric("Honeypots in top-N", sum(1 for s in top if s.features.honeypot >= 0.6))

    st.subheader(f"Top {len(top)}")
    table = []
    for rank_i, s in enumerate(top, 1):
        table.append({
            "rank": rank_i,
            "candidate_id": s.candidate_id,
            "score": round(s.score, 4),
            "title": s.features.current_title,
            "reasoning": reason_for(s, rank_i),
        })
    st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("Component breakdown (top of list)"):
        for rank_i, s in enumerate(top[:10], 1):
            c = s.components
            st.write(
                f"**#{rank_i} {s.candidate_id}** — {s.features.current_title}  ·  "
                f"score `{s.score:.3f}`  =  base `{s.base:.3f}` × avail `{s.modifier:.2f}` × penalty `{s.penalty:.2f}`  ·  "
                f"sem `{c['semantic']:.2f}` skill `{c['skill_fit']:.2f}` career `{c['career_fit']:.2f}` "
                f"exp `{c['experience']:.2f}` ctx `{c['context']:.2f}`"
                + (f"  ·  ⚠ stuffer `{s.features.keyword_stuffer:.2f}`" if s.features.keyword_stuffer >= 0.3 else "")
            )
else:
    st.info("Pick a source in the sidebar and click **Rank candidates**.")
