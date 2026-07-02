"""Grounded reasoning generator for the top-100.

Stage-4 manually reviews reasoning for: specific facts, JD connection, honest
concerns, NO hallucination, variation, and rank-consistency. We therefore build
each sentence ONLY from values that actually exist in the candidate's profile and
the components that actually drove the score — never from an LLM that could invent
a skill. Variation comes from the data itself (different facts per candidate);
tone tracks the rank band by construction.

This deliberately avoids templated name-insertion: each reasoning leads with the
candidate's real title/experience, cites the specific matched evidence, and names
the specific concern (stuffer, stale, consulting, notice period) when present.
"""

from __future__ import annotations

from .score import Scored, penalty_multiplier

_PRETTY = {
    "embeddings": "embeddings", "retrieval": "retrieval", "ranking": "ranking",
    "learning to rank": "learning-to-rank", "vector search": "vector search",
    "semantic search": "semantic search", "recommendation systems": "recsys",
    "faiss": "FAISS", "pinecone": "Pinecone", "weaviate": "Weaviate",
    "qdrant": "Qdrant", "milvus": "Milvus", "opensearch": "OpenSearch",
    "elasticsearch": "Elasticsearch", "ndcg": "NDCG", "nlp": "NLP",
    "fine-tuning llms": "LLM fine-tuning", "python": "Python",
}


def _skill_phrase(matched: list[str], limit: int = 3) -> str:
    names = [_PRETTY.get(m, m) for m in matched[:limit]]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def reason_for(s: Scored, rank: int) -> str:
    f = s.features
    yrs = f"{f.yoe:.1f} yrs" if f.yoe else "unstated experience"
    title = f.current_title or "candidate"

    # Lead clause — the concrete identity.
    lead = f"{title}, {yrs}"

    # Evidence clause — strongest real positive signal driving the rank.
    evidence = []
    if f.career_fit_score >= 0.25:
        evidence.append("career history shows hands-on retrieval/ranking/recsys work")
    sk = _skill_phrase(f.matched_skill_names)
    if sk and f.trusted_ai_skills >= 1:
        evidence.append(f"trusted skills in {sk}")
    elif sk:
        evidence.append(f"lists {sk}")
    if f.eval_literacy >= 0.5:
        evidence.append("references ranking-eval metrics (NDCG/MRR/A-B)")
    if f.product_company_ratio >= 0.5:
        evidence.append("product-company background")
    if f.geo_match:
        evidence.append("India-based / relocation-friendly")
    if not evidence:
        evidence.append("adjacent skills only, limited direct IR evidence")

    # Concern clause — honest gaps (required for Stage-4 credibility).
    _, pen_reasons = penalty_multiplier(f)
    concerns = list(pen_reasons)
    if f.last_active_gap_days > 150 and not concerns:
        concerns.append(f"last active ~{f.last_active_gap_days}d ago")
    if f.recruiter_response_rate < 0.15 and "stale" not in " ".join(concerns):
        concerns.append(f"low recruiter response rate ({f.recruiter_response_rate:.2f})")
    if f.notice_period_days >= 120:
        concerns.append(f"long notice period ({f.notice_period_days}d)")

    ev = "; ".join(evidence[:3])
    sentence = f"{lead}; {ev}."
    if concerns:
        # tone-match: high ranks frame concerns as minor, low ranks as decisive
        if rank <= 30:
            sentence += f" Minor concern: {concerns[0]}."
        else:
            sentence += f" Concern: {'; '.join(concerns[:2])}."
    # Keep it to 1–2 sentences, CSV-safe (commas are fine; the writer quotes).
    return sentence.replace("\n", " ").strip()
