# candmat — Approach Deck
### Intelligent Candidate Discovery & Ranking (Redrob Challenge)

---

## 1. The problem, restated

Rank the top-100 of **100,000** candidates for a *Senior AI Engineer (Founding
Team)* role — the way a recruiter would, by **understanding the role**, not
matching keywords. The JD is unusually explicit: it tells us what it rewards
(production retrieval/ranking/embeddings at product companies), what it rejects
(researchers, LangChain-only, title-chasers, consulting-only), and that the data
is **seeded with traps**: keyword stuffers, plain-language "Tier-5" fits, behavioral
twins, and ~80 honeypots with impossible profiles.

**Hard constraint that shapes everything:** the ranking step must run **CPU-only,
≤5 min, ≤16 GB, with no network and no hosted-LLM calls** — reproduced in a
sandboxed container. An "LLM-per-candidate" system is both forbidden and
infeasible. The constraint *is* the test: can you make a latency-quality tradeoff?

---

## 2. Design: offline-heavy, online-light

> Move all expensive "understanding" offline; keep the shipped ranker a fast,
> deterministic CPU pass over precomputed artifacts.

```
OFFLINE (no time limit)                 ONLINE  rank.py  (CPU, no net, <5 min)
────────────────────────                ─────────────────────────────────────
JD ─► structured rubric                 parse 100K JSONL
candidates ─► bge-small embeddings ──►  + semantic cosine (precomputed vectors)
              (artifacts/*.npy)         + structured rubric score
                                        + trap penalties + honeypot caps
                                        + pool-relative availability modifier
                                        ─► fuse ─► sort ─► top-100 ─► reasoning
```

The full 100K ranking runs in ~2 minutes on CPU. The only model
(`BAAI/bge-small-en-v1.5`) runs **offline**; `rank.py` ships with just NumPy.

---

## 3. Scoring — five components, then modulate

**Base fit** = weighted sum of:

| Component | Weight | What it captures |
|---|---|---|
| Semantic | 0.30 | cosine(JD, candidate career-history text) — catches Tier-5s |
| Skill fit | 0.28 | **trusted** must-have skill coverage |
| Career fit | 0.24 | retrieval/ranking/recsys evidence in role descriptions |
| Experience | 0.10 | 6–8 yr band + applied-ML share |
| Context | 0.08 | product-company + geo + eval-literacy |

**Final = base × availability × penalties.**

The decisive idea: **skill *trust*, not skill *presence***. Trust =
`endorsements × duration × proficiency`. A stuffer who lists "Pinecone, FAISS,
Embeddings" with 0 endorsements and 4 months of use earns ~0 credit; a
practitioner with 2 years and peer endorsements earns full credit.

---

## 4. Reading between the lines (the JD's actual ask)

- **Plain-language Tier-5** — never says "RAG"/"Pinecone" but the career history
  says *"built recommendation-style features in production."* → The semantic
  layer embeds **career descriptions** (weighted over the skills array), so these
  surface. Validated: a "ML Engineer @ BYJU'S" with recsys + LightGBM in their
  history ranks high despite a modest buzzword list.
- **Keyword stuffer** — full AI skill list, title "Marketing Manager", career =
  brand design at a consulting firm. → Detected via title class + untrusted
  skills + career descriptions that show no IR work (or contradict their own
  titles). Down-weighted hard.
- **Not actually available** — perfect on paper, last login 6 months ago, 5%
  response rate. → A **pool-relative** availability modifier (recency percentile,
  response rate, open-to-work) — relative because the synthetic clock makes
  everyone look stale in absolute terms.

---

## 5. Honeypot defense (the disqualifier)

> >10% honeypots in the top-100 = automatic disqualification.

We do **not** hard-code IDs. We detect the structural impossibilities that define
them:
- `proficiency == "expert"` with **0 months** of use (the canonical marker —
  validated to flag ~the documented count).
- Role tenure exceeding time actually elapsed since its start date.
- Skill duration grossly beyond total career length.
- Role durations summing to ~2× stated experience; multiple simultaneous current
  roles.

Result on our run: **0 honeypots and 0 keyword stuffers in the top-100.**

---

## 6. Reasoning that survives manual review

Stage-4 checks reasoning for specific facts, JD connection, honest concerns, **no
hallucination**, variation, and rank-consistency. We therefore build every
sentence **only from fields that exist in the candidate's profile** and from the
components that actually drove the score — never from a generative model that
could invent a skill. Concerns (stuffer, stale, consulting, long notice) are
surfaced honestly and framed by rank band.

> *"Staff Machine Learning Engineer, 7.0 yrs; career history shows hands-on
> retrieval/ranking/recsys work; trusted skills in semantic search, Pinecone and
> bm25; references ranking-eval metrics (NDCG/MRR/A-B)."*

---

## 7. Why this holds up at every stage

- **Stage 2 (scoring):** top-10/50 ordering is dominated by genuine ML/IR
  engineers with real retrieval evidence and the right seniority.
- **Stage 3 (reproduction):** CPU-only, no network, ~2 min, committed artifacts
  → reproduces unmodified. Honeypot rate 0%.
- **Stage 4 (review):** grounded, varied reasoning; real git iteration; code is
  engineering, not API calls.
- **Stage 5 (defend):** the offline/online split, the trust metric, and the
  honeypot detector are concrete design decisions we can walk through.

---

## 8. Honest limitations

- bge-small is a compact model; a larger encoder (offline) would sharpen semantic
  recall further.
- Honeypot detection is deliberately conservative (favor precision) to avoid
  demoting genuine seniors — it may not flag every planted honeypot, but it keeps
  them out of the top-100, which is what the metric rewards.
- Component weights are reasoned from the JD, not learned (no labels available);
  they are the obvious lever for future tuning if a validation signal appears.
