"""
run_eval.py
-----------
Executes eval/test_cases.py against the live graph and writes:
  - eval/results.csv          full per-case results
  - eval/results_summary.md   pass-rate table by category + weakest-category note
Run: python3 -m eval.run_eval   (from the afl_capstone/ directory)
"""
from __future__ import annotations
import sys
import csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import run_turn
from eval.test_cases import CASES


def run_case(case):
    states = []
    prior = None
    for turn in case["turns"]:
        prior = run_turn(turn, prior_state=prior)
        states.append(prior)
    try:
        passed, detail = case["check"](states)
    except Exception as e:
        passed, detail = False, f"check raised {type(e).__name__}: {e}"
    return states, passed, detail


def main():
    rows = []
    by_category = defaultdict(lambda: [0, 0])  # category -> [passed, total]

    for case in CASES:
        states, passed, detail = run_case(case)
        by_category[case["category"]][1] += 1
        by_category[case["category"]][0] += int(passed)
        rows.append({
            "id": case["id"], "category": case["category"],
            "turns": " | ".join(case["turns"]),
            "final_intent": states[-1].get("intent"),
            "final_response": states[-1]["final_response"][:140],
            "pass": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    out_csv = Path(__file__).parent / "results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total_passed = sum(v[0] for v in by_category.values())
    total_cases = sum(v[1] for v in by_category.values())

    weakest = min(by_category.items(), key=lambda kv: kv[1][0] / kv[1][1])

    lines = []
    lines.append("# AFL Assistant -- Evaluation Results\n")
    lines.append(f"**Overall pass rate: {total_passed}/{total_cases} "
                  f"({total_passed/total_cases*100:.1f}%)**\n")
    lines.append("| Category | Passed | Total | Pass rate |")
    lines.append("|---|---|---|---|")
    for cat, (p, t) in sorted(by_category.items()):
        lines.append(f"| {cat} | {p} | {t} | {p/t*100:.1f}% |")
    lines.append("")
    lines.append(f"**Weakest category: `{weakest[0]}`** "
                 f"({weakest[1][0]}/{weakest[1][1]} = {weakest[1][0]/weakest[1][1]*100:.1f}%)\n")
    lines.append("## Per-case detail\n")
    lines.append("| ID | Category | Result | Detail |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['id']} | {r['category']} | {r['pass']} | {r['detail']} |")

    out_md = Path(__file__).parent / "results_summary.md"
    out_md.write_text("\n".join(lines))

    print("\n".join(lines[:12]))
    print(f"\nFull results: {out_csv}")
    print(f"Summary: {out_md}")
    return rows, by_category


if __name__ == "__main__":
    main()
