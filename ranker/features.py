"""Per-candidate feature extraction.

Turns one raw candidate dict into a flat ``Features`` record the scorer consumes.
Everything here is deterministic, dependency-free, and reads the WHOLE profile —
skills array, career-history descriptions, education, and behavioral signals —
because the JD's traps are precisely about profiles where one part contradicts
another (AI skills on a non-tech career; expert skills with no tenure).

Design notes grounded in the profiled data (see precompute/profile_data.py):
* Skill *trust* = endorsements + duration, NOT mere presence — this is what
  separates a genuine practitioner from a keyword stuffer.
* ``github_activity_score == -1`` means "no GitHub linked", treated as unknown
  (neutral), never as a penalty (~50% of the pool has none).
* Availability is computed RELATIVE to the pool (the synthetic clock makes every
  candidate look stale in absolute terms); raw recency is captured here and
  percentile-normalised later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import jd_rubric as R
from . import text as T
from .dates import parse_date
from .honeypot import DATASET_TODAY, honeypot_score

# Built once: maps a skill NAME onto canonical rubric entries via cheap substring
# checks (replaces an O(skills x vocab) regex loop in the hot path).
_MUST_NICE_MATCHER = T.build_canonical_matcher(
    list(R.MUST_HAVE_SKILLS) + list(R.NICE_TO_HAVE_SKILLS)
)
_OFF_DOMAIN = {T.normalize(s) for s in R.OFF_DOMAIN_SKILLS}


@dataclass
class Features:
    candidate_id: str
    # role identity
    current_title: str = ""
    title_class: str = "other"            # strong | adjacent | non_engineering | other
    # skills (trusted)
    must_have_score: float = 0.0          # weighted, trust-adjusted coverage 0..1
    nice_have_score: float = 0.0
    off_domain_ratio: float = 0.0         # share of skills that are CV/speech/etc
    ai_skill_names: int = 0               # count of must-have skill *names* present
    trusted_ai_skills: int = 0            # of those, how many are trusted
    matched_skill_names: list[str] = field(default_factory=list)
    # career history evidence
    career_fit_score: float = 0.0         # 0..1 from descriptions (Tier-5 catcher)
    eval_literacy: float = 0.0            # NDCG/MRR/A-B mentions
    title_desc_mismatch: float = 0.0      # roles whose description != their title
    product_company_ratio: float = 0.0
    consulting_ratio: float = 0.0
    job_hops: int = 0                     # roles < 18 months (title-chaser signal)
    median_tenure_months: float = 0.0
    research_only: bool = False
    langchain_only: bool = False
    # experience
    yoe: float = 0.0
    applied_ml_months: float = 0.0
    # context
    geo_match: bool = False
    willing_to_relocate: bool = False
    # behavioral (raw; normalised later)
    last_active_gap_days: int = 999
    recruiter_response_rate: float = 0.0
    open_to_work: bool = False
    profile_completeness: float = 0.0
    github_activity: float = -1.0
    saved_by_recruiters_30d: int = 0
    notice_period_days: int = 90
    interview_completion_rate: float = 0.0
    # traps
    honeypot: float = 0.0
    honeypot_reasons: list[str] = field(default_factory=list)
    keyword_stuffer: float = 0.0          # 0..1


def _title_class(title: str) -> str:
    t = T.normalize(title)
    if t in R.STRONG_TITLES:
        return "strong"
    if t in R.ADJACENT_TITLES:
        return "adjacent"
    if t in R.NON_ENGINEERING_TITLES:
        return "non_engineering"
    # substring fallback for senior-prefixed variants
    if any(s in t for s in ("ml engineer", "machine learning", "ai engineer",
                            "data engineer", "data scientist", "backend",
                            "software engineer", "full stack")):
        return "strong"
    return "other"


def _skill_trust(skill: dict) -> float:
    """0..1 trust for a single skill from endorsements + duration + proficiency.

    A stuffer lists the right names with 0 endorsements and tiny/zero duration;
    a practitioner has months of use and peer endorsements.
    """
    dur = skill.get("duration_months") or 0
    end = skill.get("endorsements") or 0
    prof = skill.get("proficiency") or "beginner"
    dur_score = min(1.0, dur / 24.0)            # saturates at 2 years
    end_score = min(1.0, end / 20.0)            # saturates at 20 endorsements
    prof_score = {"beginner": 0.25, "intermediate": 0.5,
                  "advanced": 0.8, "expert": 1.0}.get(prof, 0.4)
    # expert with no duration is NOT trusted (honeypot-shaped) — floor it.
    if prof == "expert" and dur == 0:
        return 0.0
    return round(0.45 * dur_score + 0.25 * end_score + 0.30 * prof_score, 4)


def extract(candidate: dict) -> Features:
    profile = candidate.get("profile", {}) or {}
    skills = candidate.get("skills", []) or []
    career = candidate.get("career_history", []) or []
    sig = candidate.get("redrob_signals", {}) or {}

    f = Features(candidate_id=candidate.get("candidate_id", ""))
    f.current_title = profile.get("current_title", "") or ""
    f.title_class = _title_class(f.current_title)
    f.yoe = float(profile.get("years_of_experience") or 0)

    # ── Skills: trusted, weighted must-have coverage ─────────────────────────
    skill_index: dict[str, float] = {}   # canonical skill -> best trust seen
    off_domain = 0
    for s in skills:
        name = T.normalize(s.get("name", ""))
        if not name:
            continue
        trust = _skill_trust(s)
        # map this skill name onto canonical rubric entries it evidences (cheap
        # substring matching against the precomputed matcher)
        for canon in T.match_canonicals(name, _MUST_NICE_MATCHER):
            if trust > skill_index.get(canon, 0.0):
                skill_index[canon] = trust
        if name in _OFF_DOMAIN:
            off_domain += 1

    f.off_domain_ratio = round(off_domain / len(skills), 3) if skills else 0.0
    f.matched_skill_names = [c for c in skill_index if c in R.MUST_HAVE_SKILLS]
    f.ai_skill_names = len(f.matched_skill_names)
    f.trusted_ai_skills = sum(1 for c in f.matched_skill_names if skill_index[c] >= 0.45)

    must_total = sum(R.MUST_HAVE_SKILLS.values())
    must_got = sum(R.MUST_HAVE_SKILLS[c] * skill_index[c]
                   for c in skill_index if c in R.MUST_HAVE_SKILLS)
    f.must_have_score = round(min(1.0, must_got / (must_total * 0.35)), 4)  # 35% coverage ~ full credit

    nice_total = sum(R.NICE_TO_HAVE_SKILLS.values()) or 1.0
    nice_got = sum(R.NICE_TO_HAVE_SKILLS[c] * skill_index[c]
                   for c in skill_index if c in R.NICE_TO_HAVE_SKILLS)
    f.nice_have_score = round(min(1.0, nice_got / (nice_total * 0.4)), 4)

    # ── Career history: the Tier-5 catcher + product/consulting/job-hop ──────
    career_text_parts = []
    product_hits = consulting_hits = 0
    tenures = []
    title_mismatch = 0
    applied_ml_months = 0.0
    for j in career:
        desc = T.normalize(j.get("description", ""))
        jtitle = T.normalize(j.get("title", ""))
        career_text_parts.append(desc)
        dur = j.get("duration_months") or 0
        tenures.append(dur)
        if dur and dur < 18 and not j.get("is_current"):
            f.job_hops += 1
        company = T.normalize(j.get("company", ""))
        industry = T.normalize(j.get("industry", ""))
        if T.any_phrase(industry, R.PRODUCT_INDUSTRIES):
            product_hits += 1
        if T.any_phrase(company, R.CONSULTING_FIRMS) or T.any_phrase(industry, R.SERVICES_INDUSTRIES):
            consulting_hits += 1
        # title/description mismatch: description describes a clearly different role
        if jtitle and desc and _describes_other_role(jtitle, desc):
            title_mismatch += 1
        # applied-ML months: role is engineering/ML/data flavored — counts toward
        # the "4-5 years applied ML at product companies" the JD wants.
        if any(k in jtitle for k in ("ml", "machine learning", "data", "ai",
                                     "software", "backend", "engineer", "developer")):
            applied_ml_months += dur

    career_text = " ".join(career_text_parts)
    fit_hits = T.count_phrases(career_text, R.CAREER_FIT_PHRASES)
    f.career_fit_score = round(min(1.0, fit_hits / 4.0), 4)  # 4 distinct hits ~ full
    f.eval_literacy = round(min(1.0, T.count_phrases(career_text, R.EVAL_PHRASES) / 2.0), 4)
    f.title_desc_mismatch = round(title_mismatch / len(career), 3) if career else 0.0
    f.product_company_ratio = round(product_hits / len(career), 3) if career else 0.0
    f.consulting_ratio = round(consulting_hits / len(career), 3) if career else 0.0
    f.median_tenure_months = float(sorted(tenures)[len(tenures) // 2]) if tenures else 0.0
    f.applied_ml_months = applied_ml_months

    summary = T.normalize(profile.get("summary", "")) + " " + T.normalize(profile.get("headline", ""))
    f.research_only = (T.any_phrase(summary + " " + career_text, R.RESEARCH_ONLY_MARKERS)
                       and f.product_company_ratio == 0.0 and f.career_fit_score == 0.0)
    # langchain-only: LLM-wrapper vocab present but no deeper IR/ranking evidence
    f.langchain_only = (T.any_phrase(summary + " " + career_text, R.LANGCHAIN_ONLY_MARKERS)
                        and f.career_fit_score < 0.25 and f.trusted_ai_skills <= 2)

    # ── Context: geo + relocate ──────────────────────────────────────────────
    loc = T.normalize(profile.get("location", "")) + " " + T.normalize(profile.get("country", ""))
    f.geo_match = T.any_phrase(loc, R.PREFERRED_LOCATIONS) or "india" in loc
    f.willing_to_relocate = bool(sig.get("willing_to_relocate"))

    # ── Behavioral signals (raw) ─────────────────────────────────────────────
    la = parse_date(sig.get("last_active_date"))
    f.last_active_gap_days = (DATASET_TODAY - la).days if la else 999
    f.recruiter_response_rate = float(sig.get("recruiter_response_rate") or 0.0)
    f.open_to_work = bool(sig.get("open_to_work_flag"))
    f.profile_completeness = float(sig.get("profile_completeness_score") or 0.0)
    gh = sig.get("github_activity_score")
    f.github_activity = float(gh) if gh is not None else -1.0
    f.saved_by_recruiters_30d = int(sig.get("saved_by_recruiters_30d") or 0)
    f.notice_period_days = int(sig.get("notice_period_days") or 90)
    f.interview_completion_rate = float(sig.get("interview_completion_rate") or 0.0)

    # ── Traps ────────────────────────────────────────────────────────────────
    f.honeypot, f.honeypot_reasons = honeypot_score(candidate)
    f.keyword_stuffer = _keyword_stuffer_score(f)
    return f


# Role keywords used to detect a description that describes a different job than
# the role's own title (a decoy-construction artifact in stuffer/twin profiles).
_ROLE_KEYWORDS = {
    "frontend": ("react", "css", "frontend", "design system", "webpack"),
    "backend": ("api", "microservice", "backend", "database", "server"),
    "devops": ("terraform", "kubernetes", "ci/cd", "aws account", "infrastructure"),
    "qa": ("test automation", "selenium", "qa engineering", "load-testing"),
    "data": ("data pipeline", "etl", "spark", "warehouse", "airflow"),
    "ml": ("model", "training", "recommendation", "forecasting", "ml "),
    "design": ("brand", "logo", "visual", "packaging", "creative"),
    "sales": ("sales", "quota", "territory", "dealer"),
    "support": ("support agents", "tickets", "escalation", "customer-facing"),
    "mechanical": ("dfm", "prototype", "subsystem", "mechanical"),
}


def _describes_other_role(title: str, desc: str) -> bool:
    """True if the description strongly matches a role family unrelated to the title."""
    title_fam = None
    for fam, kws in _ROLE_KEYWORDS.items():
        if fam in title or any(k.strip() in title for k in kws):
            title_fam = fam
            break
    best_fam, best_hits = None, 0
    for fam, kws in _ROLE_KEYWORDS.items():
        hits = sum(1 for k in kws if k in desc)
        if hits > best_hits:
            best_fam, best_hits = fam, hits
    return bool(best_fam and best_hits >= 2 and title_fam and best_fam != title_fam)


def _keyword_stuffer_score(f: Features) -> float:
    """0..1: AI keywords present on the surface but contradicted by everything else.

    Triggers when many must-have skill *names* appear but (a) the title is
    non-engineering, and/or (b) the skills are untrusted (low duration/endorse),
    and/or (c) career descriptions show no real IR/ranking work or mismatch their
    own titles. This is the JD's central trap.
    """
    if f.ai_skill_names < 4:
        return 0.0
    surface = min(1.0, f.ai_skill_names / 8.0)
    untrusted = 1.0 - (f.trusted_ai_skills / max(1, f.ai_skill_names))
    title_bad = 1.0 if f.title_class == "non_engineering" else (0.4 if f.title_class == "other" else 0.0)
    no_career_evidence = 1.0 if f.career_fit_score < 0.25 else 0.0
    mismatch = min(1.0, f.title_desc_mismatch * 2)
    score = surface * (0.40 * untrusted + 0.30 * title_bad
                       + 0.20 * no_career_evidence + 0.10 * mismatch)
    return round(min(1.0, score), 4)
