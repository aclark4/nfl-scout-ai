"""Unit tests for qb_stats using small mock DataFrames (no network needed).

Run with:  pytest -q
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_stats  # noqa: E402
from qb_stats import (  # noqa: E402
    compare_qbs,
    score_breakdown,
    score_components,
    format_name_pbp_passer,
    generate_qb_score,
    get_qb_stats,
    safe_div,
)


def make_data(qbs):
    """Build mock versions of each nflverse dataset for the given QB specs."""
    ngs, pfr, weekly, pbp, ftn, seasonal = [], [], [], [], [], []
    for i, q in enumerate(qbs):
        name, pbp_name = q["name"], q["pbp_name"]
        # week 0 = season total; a weekly row is included to make sure we pick week 0
        ngs.append(dict(player_display_name=name, season_type="REG", week=1, attempts=30,
                        passer_rating=50.0, pass_yards=200, pass_touchdowns=1, interceptions=0,
                        completion_percentage=60.0, completion_percentage_above_expectation=0.0,
                        aggressiveness=10.0, avg_time_to_throw=2.5))
        ngs.append(dict(player_display_name=name, season_type="REG", week=0, attempts=q["att"],
                        passer_rating=100.0, pass_yards=q["yds"], pass_touchdowns=q["td"],
                        interceptions=q["ints"], completion_percentage=66.66,
                        completion_percentage_above_expectation=2.34, aggressiveness=15.2,
                        avg_time_to_throw=q["ttt"]))
        pfr.append(dict(player=name, bad_throw_pct=0.15, intended_air_yards_per_pass_attempt=8.1))
        for wk in range(1, 3):
            weekly.append(dict(pfr_player_name=name, game_type="REG", times_sacked=2, times_pressured=q["press"]))
        weekly.append(dict(pfr_player_name=name, game_type="POST", times_sacked=9, times_pressured=9))
        seasonal.append(dict(name=name, season_type="REG", rushing_yards=100, carries=20, rushing_tds=1,
                             rushing_epa=2.0, passing_epa=40.0, attempts=q["att"], sacks=20,
                             rushing_fumbles_lost=1, sack_fumbles_lost=1, dakota=0.1))
        for p in range(3):
            pbp.append(dict(passer_player_name=pbp_name, season_type="REG",
                            nflverse_game_id=f"g{i}", play_id=p))
            ftn.append(dict(nflverse_game_id=f"g{i}", nflverse_play_id=p,
                            is_interception_worthy=1 if p < q["twp"] else 0))
    return {k: pd.DataFrame(v) for k, v in dict(ngs=ngs, seasonal_pfr=pfr, weekly=weekly,
                                                  pbp=pbp, ftn=ftn, seasonal=seasonal).items()}


BURROW = dict(name="Joe Burrow", pbp_name="J.Burrow", att=652, yds=4918, td=43, ints=9, ttt=2.7, press=10, twp=2)
PENIX = dict(name="Michael Penix Jr.", pbp_name="M.Penix", att=100, yds=775, td=3, ints=0, ttt=2.4, press=0, twp=1)


@pytest.fixture
def data():
    return make_data([BURROW, PENIX])


def test_format_name_handles_suffix_and_casing():
    assert format_name_pbp_passer("Patrick Mahomes") == "P.Mahomes"
    assert format_name_pbp_passer("Michael Penix Jr.") == "M.Penix"
    assert format_name_pbp_passer("J.J. McCarthy") == "J.McCarthy"
    assert format_name_pbp_passer("Mahomes") is None


def test_names_match_across_suffix_differences():
    data = make_data([dict(BURROW, name="Gardner Minshew", pbp_name="G.Minshew")])
    data["seasonal_pfr"]["player"] = "Gardner Minshew II"
    data["weekly"]["pfr_player_name"] = "Gardner Minshew II"
    assert get_qb_stats("Gardner Minshew", data) is not None


def test_safe_div():
    assert safe_div(10, 2) == 5
    assert safe_div(10, 0) == 0.0
    assert safe_div(10, float("nan"), default=-1) == -1


def test_get_qb_stats_uses_season_total_and_regular_season(data):
    stats = get_qb_stats("joe burrow", data)  # case-insensitive
    assert stats["pass_attempts"] == 652  # week 0 row, not the weekly row
    assert stats["games_played"] == 2  # POST row excluded
    assert stats["pressure_to_sack_percentage"] == 20.0  # 4 sacks / 20 pressures
    assert stats["turnover_worthy_passes"] == 2
    assert stats["completion_percentage"] == 66.7


def test_stats_are_json_serializable(data):
    json.dumps(get_qb_stats("Joe Burrow", data))  # raises if numpy types leak through


def test_unknown_player_returns_none(data):
    assert get_qb_stats("Not A Player", data) is None


def test_zero_ints_and_zero_pressures_do_not_crash(data):
    stats = get_qb_stats("Michael Penix Jr.", data)
    assert stats is not None
    assert stats["pressure_to_sack_percentage"] == 0
    assert isinstance(generate_qb_score(stats), float)


def test_time_to_throw_positive_branch_is_applied(data):
    """Regression test: the original code computed but never assigned this value."""
    stats = get_qb_stats("Joe Burrow", data)
    slow = dict(stats, avg_time_to_throw=3.0)  # +0.5s -> (5)^2 = 25
    base = dict(stats, avg_time_to_throw=2.5)  # 0
    pts_effect = -0.15 * (slow["pressure_to_sack_percentage"] / 3.0 - base["pressure_to_sack_percentage"] / 2.5)
    diff = generate_qb_score(slow) - generate_qb_score(base)
    assert diff == pytest.approx(25 * 0.05 + pts_effect, abs=0.02)


def test_higher_pressure_to_sack_rate_lowers_score(data):
    stats = get_qb_stats("Joe Burrow", data)
    low = dict(stats, pressure_to_sack_percentage=10.0)
    high = dict(stats, pressure_to_sack_percentage=30.0)
    assert generate_qb_score(high) < generate_qb_score(low)


def test_compare_qbs(data):
    a, b = get_qb_stats("Joe Burrow", data), get_qb_stats("Michael Penix Jr.", data)
    result = compare_qbs(a, b)
    assert result["winner"] in ("Joe Burrow", "Michael Penix Jr.")
    assert result["margin"] >= 0


def test_build_writes_ranked_json(data, tmp_path, monkeypatch):
    monkeypatch.setattr(build_stats, "load_data", lambda year: data)
    out = tmp_path / "qb.json"
    result = build_stats.build(2024, min_attempts=50, out_path=out)
    saved = json.loads(out.read_text())
    assert saved["player_count"] == 2 == result["player_count"]
    assert [p["rank"] for p in saved["players"]] == [1, 2]
    assert saved["players"][0]["score"] >= saved["players"][1]["score"]


def test_min_attempts_filter(data):
    assert build_stats.qualifying_qbs(data, 200) == ["Joe Burrow"]


def test_components_sum_to_score(data):
    stats = get_qb_stats("Joe Burrow", data)
    assert round(sum(score_components(stats).values()), 2) == generate_qb_score(stats)
    assert set(score_breakdown(stats)) == set(score_components(stats))


def test_compare_breakdown_explains_margin(data):
    a, b = get_qb_stats("Joe Burrow", data), get_qb_stats("Michael Penix Jr.", data)
    result = compare_qbs(a, b)
    points = [e["points"] for e in result["breakdown"]]
    assert points == sorted(points, reverse=True)
    net = sum(e[a["name"]] - e[b["name"]] for e in result["breakdown"])
    assert abs(abs(net) - result["margin"]) < 0.05
    top = result["breakdown"][0]
    assert top["favors"] in (a["name"], b["name"])
