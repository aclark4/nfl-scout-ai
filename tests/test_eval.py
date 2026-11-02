"""Tests for the evaluation grader and report (no API calls)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import QUESTIONS_PATH, grade, normalize, summarize  # noqa: E402


def test_normalize_handles_commas_and_apostrophes():
    assert normalize("Burrow threw for 4,918 yards") == "burrow threw for 4918 yards"
    assert normalize("Ja’Marr Chase") == "ja'marr chase"
    assert normalize("0.32, then 1,000,000") == "0.32, then 1000000"


def test_grade_must_include_and_tools():
    case = {"must_include": ["4918"], "expected_tool": "get_player_stats"}
    good = grade(case, "He threw for 4,918 yards.", ["get_player_stats"], tools_enabled=True)
    assert good["correct"] and good["tool_ok"]
    bad = grade(case, "He threw for about 5,000 yards.", [], tools_enabled=True)
    assert not bad["correct"] and bad["missing"] == ["4918"] and bad["tool_ok"] is False
    no_tools = grade(case, "4918", [], tools_enabled=False)
    assert no_tools["tool_ok"] is None


def test_grade_refusals_and_forbidden():
    case = {"any_of": ["not in", "don't have"], "must_not_include": ["Ja'Marr Chase"]}
    assert grade(case, "Receivers are not in this dataset.", [], True)["correct"]
    assert not grade(case, "Ja’Marr Chase led the league.", [], True)["correct"]


def test_questions_file_is_valid():
    cases = json.loads(QUESTIONS_PATH.read_text())
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["category"] in {"lookup", "comparison", "ranking", "methodology", "out_of_scope"}
        assert c.get("must_include") or c.get("any_of")


def test_summary_has_tables():
    row = dict(mode="rag_tools", id="q1", category="lookup", correct=True, tool_ok=True, latency_ms=2000,
               usage={"input_tokens": 100, "output_tokens": 20}, cost_usd=None, missing=[],
               forbidden_found=[], any_of_ok=True, error=None)
    fail = dict(row, id="q2", correct=False, missing=["43"])
    md = summarize({"model": "m", "run_at": "now", "rows": [row, fail]})
    assert "| rag_tools | 1/2 (50%) | 2/2 (100%) | 2.0 s | 120 |" in md
    assert "`rag_tools` / `q2`: missing ['43']" in md
