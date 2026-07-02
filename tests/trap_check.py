#!/usr/bin/env python3
"""Trap sanity harness — our own mini ground-truth, since the real one is hidden.

Verifies that the ranker behaves correctly on the documented trap archetypes:
  * honeypots (expert-with-0-months / impossible dates)  -> must NOT reach top 100
  * keyword stuffers (AI skills + non-tech title)        -> must rank poorly
  * genuine Tier-5 (recsys in career, product company)   -> must rank well
  * honeypot rate in the produced top-100                -> must be < 10%

Run AFTER producing submission.csv:
    python tests/trap_check.py --candidates "<path>/candidates.jsonl" --submission submission.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.features import extract
from ranker.honeypot import is_honeypot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", default="submission.csv")
    args = ap.parse_args()

    top_ids = []
    with open(args.submission, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            top_ids.append(row["candidate_id"])
    top_set = set(top_ids)
    print(f"top-100 loaded: {len(top_ids)} ids")

    honeypots = []
    stuffers = []
    tier5 = []
    top_honeypots = []
    by_id_rank = {cid: i + 1 for i, cid in enumerate(top_ids)}

    with open(args.candidates, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c["candidate_id"]
            if is_honeypot(c):
                honeypots.append(cid)
                if cid in top_set:
                    top_honeypots.append(cid)
            fe = extract(c)
            if fe.keyword_stuffer >= 0.5:
                stuffers.append(cid)
            if (fe.career_fit_score >= 0.5 and fe.title_class == "strong"
                    and fe.product_company_ratio >= 0.5):
                tier5.append((cid, by_id_rank.get(cid)))

    n_top = len(top_ids)
    hp_rate = len(top_honeypots) / n_top if n_top else 0
    stuffers_in_top = [c for c in stuffers if c in top_set]
    tier5_in_top = [(c, rk) for c, rk in tier5 if rk]

    print("\n=== TRAP CHECK ===")
    print(f"honeypots detected in pool      : {len(honeypots)}")
    print(f"honeypots in top-100            : {len(top_honeypots)}  (rate {hp_rate:.1%})  "
          f"{'OK' if hp_rate < 0.10 else 'FAIL >10% DISQUALIFY'}")
    print(f"keyword stuffers in pool        : {len(stuffers)}")
    print(f"keyword stuffers in top-100     : {len(stuffers_in_top)}  "
          f"{'OK' if len(stuffers_in_top) <= 5 else 'WARN'}")
    print(f"strong Tier-5 fits detected     : {len(tier5)}")
    print(f"  of those, in top-100          : {len(tier5_in_top)}  "
          f"(e.g. {tier5_in_top[:5]})")

    ok = hp_rate < 0.10
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
