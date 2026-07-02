#!/usr/bin/env python3
"""One-off dataset profiler — grounds feature/honeypot/penalty design in real data.

Streams candidates.jsonl (low memory) and reports the distributions that drive the
ranker: titles, industries, company sizes, signal envelopes, career-history shapes,
and candidate evidence for the documented traps (keyword stuffers, Tier-5s, honeypots).

Run:  python precompute/profile_data.py --candidates "<path>/candidates.jsonl"
Not part of the shipped ranking step — analysis only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date

# Skill/title vocab the JD cares about, for quick trap reconnaissance.
AI_SKILL_HINTS = {
    "embeddings", "retrieval", "ranking", "rag", "vector", "pinecone", "weaviate",
    "qdrant", "milvus", "faiss", "opensearch", "elasticsearch", "bm25", "ndcg",
    "sentence-transformers", "bge", "e5", "fine-tuning llms", "lora", "qlora",
    "nlp", "information retrieval", "recommendation", "recsys", "learning to rank",
}
NON_TECH_TITLES = {
    "marketing manager", "hr manager", "sales executive", "accountant",
    "content writer", "graphic designer", "civil engineer", "mechanical engineer",
    "operations manager", "customer support", "project manager", "business analyst",
}
CONSULTING = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
              "tata consultancy", "hcl", "tech mahindra"}


def _parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    n = 0
    titles = Counter()
    industries = Counter()
    company_sizes = Counter()
    edu_tiers = Counter()
    work_modes = Counter()
    proficiencies = Counter()

    yoe = []
    skills_per = []
    careers_per = []
    completeness = []
    resp_rate = []
    last_active_gap_days = []
    github = []
    notice = []

    # trap recon counters
    stuffer_suspects = 0      # >=6 AI skills but non-tech current title
    tier5_suspects = 0        # recsys/ranking in career desc but few AI skill names
    zero_dur_expert = 0       # "expert" proficiency w/ 0 duration_months (honeypot marker)
    tenure_gt_company_age = 0 # crude: duration_months implies start before plausible
    consulting_only = 0
    inactive_perfect = 0      # high completeness but stale + low response

    today = date(2026, 6, 12)

    examples = {"stuffer": None, "tier5": None, "zero_dur_expert": None}

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            if args.limit and n > args.limit:
                n -= 1
                break
            c = json.loads(line)
            prof = c.get("profile", {})
            sig = c.get("redrob_signals", {})
            ch = c.get("career_history", []) or []
            skills = c.get("skills", []) or []
            edu = c.get("education", []) or []

            cur_title = (prof.get("current_title") or "").strip().lower()
            titles[cur_title] += 1
            industries[(prof.get("current_industry") or "").strip()] += 1
            company_sizes[prof.get("current_company_size") or ""] += 1
            for e in edu:
                edu_tiers[e.get("tier") or "unknown"] += 1

            yoe.append(prof.get("years_of_experience") or 0)
            skills_per.append(len(skills))
            careers_per.append(len(ch))
            completeness.append(sig.get("profile_completeness_score") or 0)
            resp_rate.append(sig.get("recruiter_response_rate") or 0)
            github.append(sig.get("github_activity_score"))
            notice.append(sig.get("notice_period_days"))
            work_modes[sig.get("preferred_work_mode") or ""] += 1

            la = _parse_date(sig.get("last_active_date"))
            gap = (today - la).days if la else None
            if gap is not None:
                last_active_gap_days.append(gap)

            skill_names = {(s.get("name") or "").strip().lower() for s in skills}
            for s in skills:
                proficiencies[s.get("proficiency") or ""] += 1
                if (s.get("proficiency") == "expert") and not s.get("duration_months"):
                    zero_dur_expert += 1
                    if examples["zero_dur_expert"] is None:
                        examples["zero_dur_expert"] = c.get("candidate_id")

            ai_hits = sum(1 for s in skill_names if any(h in s for h in AI_SKILL_HINTS))
            career_text = " ".join((j.get("description") or "").lower() for j in ch)
            recsys_in_career = any(k in career_text for k in
                                   ("recommend", "ranking", "retrieval", "search system",
                                    "embedding", "vector", "personaliz"))

            if ai_hits >= 6 and cur_title in NON_TECH_TITLES:
                stuffer_suspects += 1
                if examples["stuffer"] is None:
                    examples["stuffer"] = c.get("candidate_id")
            if recsys_in_career and ai_hits <= 3:
                tier5_suspects += 1
                if examples["tier5"] is None:
                    examples["tier5"] = c.get("candidate_id")

            companies = {(j.get("company") or "").strip().lower() for j in ch}
            companies |= {(prof.get("current_company") or "").strip().lower()}
            if companies and all(any(con in co for con in CONSULTING) for co in companies if co):
                consulting_only += 1

            # honeypot crude: tenure longer than the gap since earliest plausible start
            for j in ch:
                sd = _parse_date(j.get("start_date"))
                dm = j.get("duration_months") or 0
                if sd and dm:
                    months_since_start = (today.year - sd.year) * 12 + (today.month - sd.month)
                    if dm - months_since_start > 6:  # claims more tenure than time elapsed
                        tenure_gt_company_age += 1
                        break

            comp = sig.get("profile_completeness_score") or 0
            if comp >= 85 and gap is not None and gap > 150 and (sig.get("recruiter_response_rate") or 0) < 0.1:
                inactive_perfect += 1

    def stats(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        if not xs:
            return "n/a"
        xs.sort()
        q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
        return f"min={xs[0]:.1f} p25={q(.25):.1f} med={q(.5):.1f} p75={q(.75):.1f} p95={q(.95):.1f} max={xs[-1]:.1f}"

    print(f"\n===== PROFILED {n} candidates =====")
    print("\n-- current_title (top 25) --")
    for t, c in titles.most_common(25):
        print(f"  {c:6d}  {t}")
    print(f"\n-- distinct titles: {len(titles)} --")
    print("\n-- current_industry (top 20) --")
    for t, c in industries.most_common(20):
        print(f"  {c:6d}  {t}")
    print("\n-- company sizes --");  [print(f"  {c:6d}  {k}") for k, c in company_sizes.most_common()]
    print("\n-- edu tiers --");      [print(f"  {c:6d}  {k}") for k, c in edu_tiers.most_common()]
    print("\n-- skill proficiency --");[print(f"  {c:6d}  {k}") for k, c in proficiencies.most_common()]
    print("\n-- work modes --");     [print(f"  {c:6d}  {k}") for k, c in work_modes.most_common()]

    print("\n-- numeric envelopes --")
    print(f"  years_of_experience : {stats(yoe)}")
    print(f"  skills_per_candidate: {stats(skills_per)}")
    print(f"  careers_per_cand    : {stats(careers_per)}")
    print(f"  completeness_score  : {stats(completeness)}")
    print(f"  recruiter_resp_rate : {stats(resp_rate)}")
    print(f"  last_active_gap_days: {stats(last_active_gap_days)}")
    print(f"  github_activity     : {stats(github)}")
    print(f"  notice_period_days  : {stats(notice)}")

    print("\n-- TRAP RECON (heuristic counts, not ground truth) --")
    print(f"  keyword-stuffer suspects (>=6 AI skills + non-tech title): {stuffer_suspects}  e.g. {examples['stuffer']}")
    print(f"  Tier-5 suspects (recsys in career, <=3 AI skill names)   : {tier5_suspects}  e.g. {examples['tier5']}")
    print(f"  zero-duration 'expert' skills (honeypot marker)          : {zero_dur_expert}  e.g. {examples['zero_dur_expert']}")
    print(f"  tenure > time-elapsed (impossible date honeypot)         : {tenure_gt_company_age}")
    print(f"  consulting-only careers                                  : {consulting_only}")
    print(f"  inactive-but-perfect (compl>=85, stale>150d, resp<0.1)   : {inactive_perfect}")


if __name__ == "__main__":
    main()
