"""Tiny date helpers (no dependencies)."""

from __future__ import annotations

from datetime import date


def parse_date(s) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


def months_between(start: date | None, end: date | None) -> int:
    if not start or not end:
        return 0
    return max(0, (end.year - start.year) * 12 + (end.month - start.month))
