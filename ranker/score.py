"""Deterministic scoring from extracted features.

Produces a base fit score (semantic + skill + career + experience + context),
then applies a multiplicative availability modifier and trap penalties. The
weights live in ``jd_rubric.WEIGHTS``. No network, no LLM — pure arithmetic over
the precomputed features, so the whole 100K pool scores in well under the budget.

Scoring philosophy is ported from verdix (credit transferable evidence, reward
substance over keyword surface, penalise contradiction): a candidate who shows
the work in their career history outranks one who only lists the words.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import jd_rubric as R
from .features import Features


@dataclass
class Scored:
    candidate_id: str
    score: float                  # final 0..1
    base: float
    components: dict
    modifier: float
    penalty: float
    features: Features


def availability_modifier(f: Features, recency_pct: float) -> float:
    """0.55..1.10 multiplier from behavioral signals.

    The JD: "a perfect-on-paper candidate who hasn't logged in for 6 months and
    has a 5% response rate is, for hiring purposes, not actually available."
    Recency is passed in as a POOL PERCENTILE (1=most recent) because the
    synthetic clock makes every absolute gap look stale.
    """
    # response rate: full credit at >=0.5, linear below
    resp = min(1.0, f.recruiter_response_rate / 0.5)
    avail = 1.0 if f.open_to_work else 0.55
    completeness = min(1.0, f.profile_completeness / 80.0)
    saved = min(1.0, f.saved_by_recruiters_30d / 5.0)

    quality = (0.34 * recency_pct + 0.30 * resp + 0.18 * avail
               + 0.10 * completeness + 0.08 * saved)
    # map 0..1 quality onto 0.55..1.10 (good availability gives a small boost)
    return round(0.55 + 0.55 * quality, 4)


def penalty_multiplier(f: Features) -> tuple[float, list[str]]:
    """<=1.0 multiplier from disqualifier signals. Returns (mult, reasons)."""
    mult = 1.0
    reasons: list[str] = []

    if f.honeypot >= 0.6:
        mult *= 0.02
        reasons.append("honeypot/impossible profile")
    if f.keyword_stuffer >= 0.5:
        mult *= (1.0 - 0.6 * f.keyword_stuffer)
        reasons.append(f"keyword stuffer ({f.keyword_stuffer:.2f})")
    if f.title_class == "non_engineering":
        mult *= 0.45
        reasons.append(f"non-engineering current title ({f.current_title})")
    if f.consulting_ratio >= 0.8:
        mult *= 0.55
        reasons.append("career almost entirely consulting/services firms")
    if f.research_only:
        mult *= 0.5
        reasons.append("research-only, no production deployment")
    if f.langchain_only:
        mult *= 0.55
        reasons.append("LangChain/LLM-wrapper only, shallow IR depth")
    if f.job_hops >= 4:
        mult *= 0.8
        reasons.append(f"title-chaser pattern ({f.job_hops} short stints)")
    if f.off_domain_ratio >= 0.6 and f.career_fit_score < 0.25:
        mult *= 0.7
        reasons.append("primarily CV/speech/design, little NLP/IR")
    return round(mult, 4), reasons


def base_score(f: Features, semantic: float) -> tuple[float, dict]:
    """Weighted combination of the positive fit components (0..1)."""
    # experience band score: peak inside 6-8, taper outside, floor not zero
    yoe = f.yoe
    if R.TARGET_YOE_LOW <= yoe <= R.TARGET_YOE_HIGH:
        exp = 1.0
    elif yoe < R.TARGET_YOE_LOW:
        exp = max(0.3, yoe / R.TARGET_YOE_LOW)
    else:
        exp = max(0.4, 1.0 - (yoe - R.TARGET_YOE_HIGH) / 10.0)
    applied_share = min(1.0, (f.applied_ml_months / 12.0) / max(1.0, f.yoe))
    experience = round(0.6 * exp + 0.4 * applied_share, 4)

    title_bonus = {"strong": 1.0, "adjacent": 0.65,
                   "other": 0.4, "non_engineering": 0.15}.get(f.title_class, 0.4)
    skill_fit = round(min(1.0, 0.7 * f.must_have_score + 0.3 * f.nice_have_score), 4)
    # career fit blends description evidence with eval literacy and title quality
    career_fit = round(min(1.0, 0.7 * f.career_fit_score
                           + 0.18 * f.eval_literacy + 0.12 * title_bonus), 4)
    context = round(min(1.0, 0.5 * f.product_company_ratio
                        + 0.2 * (1.0 if f.geo_match else 0.0)
                        + 0.2 * f.eval_literacy
                        + 0.1 * (1.0 if f.github_activity > 20 else 0.0)), 4)

    comps = {
        "semantic": round(semantic, 4),
        "skill_fit": skill_fit,
        "career_fit": career_fit,
        "experience": experience,
        "context": context,
    }
    base = sum(R.WEIGHTS[k] * comps[k] for k in R.WEIGHTS)
    return round(base, 6), comps


def score_candidate(f: Features, semantic: float, recency_pct: float) -> Scored:
    base, comps = base_score(f, semantic)
    modifier = availability_modifier(f, recency_pct)
    penalty, _ = penalty_multiplier(f)
    final = round(base * modifier * penalty, 6)
    return Scored(candidate_id=f.candidate_id, score=final, base=base,
                  components=comps, modifier=modifier, penalty=penalty, features=f)
