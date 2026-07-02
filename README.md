# candmat — Redrob Intelligent Candidate Discovery & Ranking

[![rank-and-validate](https://github.com/StarkPrince/candmat/actions/workflows/rank.yml/badge.svg)](https://github.com/StarkPrince/candmat/actions/workflows/rank.yml)

> CI runs `rank.py` on a committed 200-candidate fixture (CPU-only, no network),
> validates the output against the official spec validator, and asserts 0
> honeypots / 0 keyword stuffers in the produced top-100 — on every push.

Ranks the top-100 candidates from a 100,000-profile pool against the released
*Senior AI Engineer (Founding Team)* job description — the way a recruiter would,
by reading the whole profile rather than matching keywords.

## Approach (short)

An **offline-heavy, online-light hybrid**. Expensive understanding is precomputed;
the shipped ranking step is a deterministic CPU pass that fits the contest budget
(**≤5 min, ≤16 GB, CPU-only, no network/LLM at rank time**).

1. **Semantic recall (offline)** — a small local sentence-transformer
   (`BAAI/bge-small-en-v1.5`) embeds the JD and each candidate's *career-history
   substance* (descriptions, not just the skills array). Saved to
   `artifacts/embeddings.npy`. At rank time we only do a NumPy cosine — no model,
   no network. This surfaces the **"Tier-5"** candidates who built a recsys/search
   system but never wrote "RAG" or "Pinecone".
2. **Structured fit (rank time)** — weighted scoring against a JD rubric
   (`ranker/jd_rubric.py`): trusted must-have skills, career-history evidence,
   experience band, product-company + geo + eval-literacy context.
   Skill **trust = endorsements × duration × proficiency**, so a keyword stuffer
   who lists the right words with zero tenure earns no credit.
3. **Trap defense** — explicit detectors for the documented traps:
   - **Honeypots** (`ranker/honeypot.py`): "expert" skills with 0 months used,
     tenure exceeding elapsed time, durations beyond total experience → forced to
     ~0. (Validated: catches ~84 profiles, matching the documented ~80.)
   - **Keyword stuffers**: many AI skill *names* but non-engineering title,
     untrusted skills, or career descriptions with no real IR work / that
     contradict their own title.
   - **Inactivity**: a pool-relative availability modifier down-weights stale,
     low-response, not-open-to-work profiles ("not actually hireable").
4. **Fusion & output** — `base × availability × penalties`, sorted by score
   desc then `candidate_id` asc (the spec's tie-break), top 100, each with a
   1–2 sentence reasoning built **only** from real profile facts (no LLM, no
   hallucination), then self-validated against `validate_submission.py`.

No hosted LLM is called anywhere in the ranking path. (An optional *offline* LLM
pass to polish reasoning prose is possible but not required and is off by default.)

## Reproduce

```bash
pip install -r requirements.txt

# 1) (optional, offline) regenerate semantic embeddings — needs the heavy deps.
#    Skippable: committed artifacts/ already contain them. Exceeds the 5-min
#    window by design; the RANKING step below does not.
pip install -r requirements-offline.txt
python precompute/embed_candidates.py --candidates ./candidates.jsonl

# 2) produce the submission (this is the single Stage-3 reproduce command;
#    CPU-only, no network, < 5 min on 16 GB)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`rank.py` runs even without `artifacts/` (semantic component falls back to 0 and
the rule/career/signal layers still produce a valid ranking), so the repo is
runnable on a clone before embeddings are generated.

## Verify

```bash
python validate_submission.py submission.csv         # format gate
python tests/trap_check.py --candidates ./candidates.jsonl --submission submission.csv
```

## Layout

```
rank.py                 single reproduce command (CPU, no net)
ranker/
  jd_rubric.py          JD distilled into weighted skills/traps/geo
  text.py               alias-aware skill-evidence matching
  dates.py              date helpers
  features.py           per-candidate feature extraction (reads whole profile)
  honeypot.py           impossibility detector
  score.py              base score + availability modifier + penalties
  reasoning.py          grounded, varied, rank-consistent reasoning
precompute/
  profile_data.py       dataset profiler (analysis only)
  embed_candidates.py   OFFLINE local embeddings -> artifacts/
artifacts/              precomputed embeddings (committed)
tests/trap_check.py     trap sanity harness
validate_submission.py  vendored spec validator
```
