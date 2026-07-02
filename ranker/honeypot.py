"""Honeypot + impossibility detection.

The dataset embeds ~80 honeypots with "subtly impossible" profiles, forced to
relevance tier 0 in the ground truth. Ranking one in the top 100 hurts; >10% in
the top 100 is an automatic Stage-3 disqualification. We do NOT special-case
known IDs (the spec says a good system avoids them naturally) — we detect the
structural impossibilities that define them.

Validated against the data: ``proficiency=='expert' & duration_months==0`` alone
flags ~84 skills, matching the documented ~80 honeypots. We combine several
independent impossibility checks and return a 0..1 honeypot score; the fuser
applies a hard cap when it crosses a threshold.
"""

from __future__ import annotations

from datetime import date

from .dates import parse_date

# A reference "today" for the synthetic dataset (latest activity ~2026-05).
DATASET_TODAY = date(2026, 6, 1)


def honeypot_score(candidate: dict) -> tuple[float, list[str]]:
    """Return (score 0..1, reasons). Higher = more likely an impossible profile."""
    reasons: list[str] = []
    hits = 0.0

    profile = candidate.get("profile", {}) or {}
    skills = candidate.get("skills", []) or []
    career = candidate.get("career_history", []) or []

    # 1. "expert" proficiency with zero months of use — the canonical marker.
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and not s.get("duration_months")
    )
    if expert_zero >= 1:
        hits += min(1.0, 0.6 + 0.2 * expert_zero)
        reasons.append(f"{expert_zero} 'expert' skill(s) with 0 months used")

    # 2. Skill duration GROSSLY exceeds the candidate's career length. A skill can
    #    legitimately span overlapping roles, so only an egregious overrun (well
    #    beyond total experience) is impossible — a modest excess is normal and
    #    must NOT flag genuine seniors.
    yoe_months = (profile.get("years_of_experience") or 0) * 12
    if yoe_months:
        for s in skills:
            dm = s.get("duration_months") or 0
            if dm > yoe_months * 1.5 + 24:  # >50% beyond total experience AND +2yr
                hits += 0.5
                reasons.append(
                    f"skill '{s.get('name')}' used {dm}mo >> {yoe_months:.0f}mo total experience"
                )
                break

    # 3. Tenure claimed beyond time actually elapsed since the role start.
    for j in career:
        sd = parse_date(j.get("start_date"))
        dm = j.get("duration_months") or 0
        if sd and dm:
            elapsed = (DATASET_TODAY.year - sd.year) * 12 + (DATASET_TODAY.month - sd.month)
            if dm - elapsed > 6:
                hits += 0.5
                reasons.append(
                    f"role at '{j.get('company')}' claims {dm}mo but only ~{elapsed}mo elapsed"
                )
                break

    # 4. Sum of role durations wildly exceeds stated years of experience.
    total_role_months = sum((j.get("duration_months") or 0) for j in career)
    if yoe_months and total_role_months > yoe_months * 1.8 + 24:
        hits += 0.4
        reasons.append(
            f"career roles sum to {total_role_months}mo vs {yoe_months:.0f}mo experience"
        )

    # 5. Overlapping current roles (two is_current with incompatible spans) or
    #    career description that names a different role than its own title — a
    #    softer signal handled in features (title/description mismatch), kept here
    #    only as an impossibility when multiple roles claim is_current.
    current_count = sum(1 for j in career if j.get("is_current"))
    if current_count >= 2:
        hits += 0.3
        reasons.append(f"{current_count} simultaneous current roles")

    return min(1.0, hits), reasons


def is_honeypot(candidate: dict, threshold: float = 0.6) -> bool:
    score, _ = honeypot_score(candidate)
    return score >= threshold
