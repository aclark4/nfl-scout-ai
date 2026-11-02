"""
run_eval.py - Measure the scouting agent against questions with known answers.

Runs every question in eval/questions.json in two modes:
  rag_only   - retrieved documents only, no tools (baseline)
  rag_tools  - retrieved documents + exact-data tools (the full agent)

Grades each answer automatically, then writes:
  eval/results/<timestamp>.json   full per-question results
  eval/results/latest.md          summary tables for the README

Usage (from the project root):
    python eval/run_eval.py
    python eval/run_eval.py --modes rag_tools --only rank
    python eval/run_eval.py --price-in 1.00 --price-out 5.00   # $ per million tokens, to estimate cost
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTIONS_PATH = ROOT / "eval" / "questions.json"
RESULTS_DIR = ROOT / "eval" / "results"
MODES = {"rag_only": False, "rag_tools": True}


# ---------------------------------------------------------------- grading

def normalize(text: str) -> str:
    """Lowercase, unify apostrophes, and drop thousands separators so '4,918' matches '4918'."""
    text = text.lower().replace("’", "'")
    return re.sub(r"(?<=\d),(?=\d{3})", "", text)


def grade(case: dict[str, Any], answer: str, tools_used: list[str], tools_enabled: bool) -> dict[str, Any]:
    """Return pass/fail plus the reasons, so failures are easy to inspect."""
    text = normalize(answer)
    missing = [s for s in case.get("must_include", []) if normalize(s) not in text]
    forbidden = [s for s in case.get("must_not_include", []) if normalize(s) in text]
    any_of = case.get("any_of", [])
    any_ok = (not any_of) or any(normalize(s) in text for s in any_of)

    expected_tool = case.get("expected_tool")
    tool_ok: Optional[bool] = None
    if tools_enabled and expected_tool:
        tool_ok = expected_tool in tools_used

    return {
        "correct": not missing and not forbidden and any_ok,
        "missing": missing,
        "forbidden_found": forbidden,
        "any_of_ok": any_ok,
        "tool_ok": tool_ok,
    }


# ---------------------------------------------------------------- running

def run(modes: list[str], only: Optional[str], price_in: Optional[float], price_out: Optional[float]) -> dict:
    from dotenv import load_dotenv

    from agent import ScoutAgent
    from retriever import Retriever
    from tools import StatsStore

    load_dotenv(ROOT / ".env")
    cases = json.loads(QUESTIONS_PATH.read_text())
    if only:
        cases = [c for c in cases if c["category"] == only or c["id"] == only]

    retriever, store = Retriever(ROOT / "data" / "chroma"), StatsStore()
    rows: list[dict[str, Any]] = []

    for mode in modes:
        agent = ScoutAgent(retriever=retriever, store=store, use_tools=MODES[mode])
        print(f"\n=== {mode} ({len(cases)} questions, model={agent.model}) ===")
        for case in cases:
            try:
                result = agent.ask(case["question"])
                error = None
            except Exception as exc:  # record API errors as failures instead of stopping
                result = {"answer": "", "tools_used": [], "latency_ms": 0,
                          "usage": {"input_tokens": 0, "output_tokens": 0}, "sources": []}
                error = str(exc)
            tools = [t["name"] for t in result["tools_used"]]
            g = grade(case, result["answer"], tools, MODES[mode])
            cost = None
            if price_in is not None and price_out is not None:
                u = result["usage"]
                cost = (u["input_tokens"] * price_in + u["output_tokens"] * price_out) / 1_000_000
            rows.append({
                "mode": mode, "id": case["id"], "category": case["category"], "question": case["question"],
                "answer": result["answer"], "tools": tools, "sources": result["sources"],
                "latency_ms": result["latency_ms"], "usage": result["usage"], "cost_usd": cost,
                "error": error, **g,
            })
            mark = "PASS" if g["correct"] else "FAIL"
            tool_note = "" if g["tool_ok"] is None else f" tool={'ok' if g['tool_ok'] else 'missed'}"
            print(f"  {mark}  {case['id']:<28} {result['latency_ms']:>6} ms{tool_note}"
                  + (f"  missing={g['missing']}" if g["missing"] else "")
                  + (f"  ERROR: {error[:80]}" if error else ""))

    return {"model": agent.model, "run_at": datetime.now().isoformat(timespec="seconds"), "rows": rows}


# ---------------------------------------------------------------- reporting

def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def summarize(results: dict) -> str:
    rows = results["rows"]
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    categories = list(dict.fromkeys(r["category"] for r in rows))
    has_cost = any(r["cost_usd"] is not None for r in rows)

    lines = ["## Evaluation results", "",
             f"Model: `{results['model']}` · Run: {results['run_at']} · "
             f"{len(rows) // max(len(modes), 1)} questions per mode", ""]

    header = "| Mode | Accuracy | Correct tool used | Median latency | Avg tokens / question |"
    sep = "|---|---|---|---|---|"
    if has_cost:
        header += " Avg cost / question |"
        sep += "---|"
    lines += [header, sep]
    for mode in modes:
        mr = [r for r in rows if r["mode"] == mode]
        tool_rows = [r for r in mr if r["tool_ok"] is not None]
        tokens = statistics.mean(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in mr)
        line = (f"| {mode} | {pct(sum(r['correct'] for r in mr), len(mr))} | "
                f"{pct(sum(r['tool_ok'] for r in tool_rows), len(tool_rows)) if tool_rows else 'n/a'} | "
                f"{statistics.median(r['latency_ms'] for r in mr) / 1000:.1f} s | {tokens:,.0f} |")
        if has_cost:
            line += f" ${statistics.mean(r['cost_usd'] or 0 for r in mr):.4f} |"
        lines.append(line)

    lines += ["", "**Accuracy by category**", "",
              "| Category | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]
    for cat in categories:
        cells = []
        for mode in modes:
            cr = [r for r in rows if r["mode"] == mode and r["category"] == cat]
            cells.append(pct(sum(r["correct"] for r in cr), len(cr)))
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")

    failures = [r for r in rows if not r["correct"]]
    if failures:
        lines += ["", "**Failures**", ""]
        for r in failures:
            why = []
            if r["missing"]:
                why.append(f"missing {r['missing']}")
            if r["forbidden_found"]:
                why.append(f"contained {r['forbidden_found']}")
            if not r["any_of_ok"]:
                why.append("did not decline")
            if r["error"]:
                why.append(f"error: {r['error'][:60]}")
            lines.append(f"- `{r['mode']}` / `{r['id']}`: {'; '.join(why)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the NFL scouting agent.")
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--only", help="Run one category (lookup, comparison, ranking, methodology, out_of_scope) or question id")
    parser.add_argument("--price-in", type=float, help="USD per million input tokens (optional)")
    parser.add_argument("--price-out", type=float, help="USD per million output tokens (optional)")
    args = parser.parse_args()

    results = run(args.modes, args.only, args.price_in, args.price_out)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}.json").write_text(json.dumps(results, indent=2))
    summary = summarize(results)
    (RESULTS_DIR / "latest.md").write_text(summary)
    print("\n" + summary)
    print(f"Saved eval/results/{stamp}.json and eval/results/latest.md")


if __name__ == "__main__":
    main()
