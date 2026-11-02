"""
qb_stats.py - Reusable QB stat extraction and scoring for the 2024 NFL season.

Refactored from the original QB-Comparison-Tool CLI so the same logic can be
imported by a build script, a FastAPI service, or an AI agent tool without
triggering data downloads or interactive prompts at import time.

Usage:
    from qb_stats import load_data, get_qb_stats, generate_qb_score

    data = load_data(2024)
    stats = get_qb_stats("Joe Burrow", data)
    if stats:
        print(generate_qb_score(stats))
"""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(year: int = 2024) -> dict[str, pd.DataFrame]:
    """Download every nflverse dataset the scoring formula needs for one season.

    This is slow (play-by-play is large), so call it once and reuse the result.
    """
    import nfl_data_py as nfl  # imported here so tests can run without it

    print(f"Loading {year} NFL QB data (play-by-play takes a while)...")
    seasonal = nfl.import_seasonal_data(years=[year])
    player_ids = nfl.import_ids()
    seasonal = seasonal.merge(
        player_ids[["gsis_id", "name"]],
        left_on="player_id",
        right_on="gsis_id",
        how="left",
    )

    data = {
        "ngs": nfl.import_ngs_data(stat_type="passing", years=[year]),
        "seasonal_pfr": nfl.import_seasonal_pfr(years=[year], s_type="pass"),
        "weekly": nfl.import_weekly_pfr(years=[year], s_type="pass"),
        "pbp": nfl.import_pbp_data(years=[year]),
        "ftn": nfl.import_ftn_data(years=[year]),
        "seasonal": seasonal,
    }
    print("Data loaded!")
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning `default` instead of crashing on a zero/NaN denominator."""
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def to_python(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python types so results are JSON-safe."""
    if value is None:
        return None
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def format_name_pbp_passer(qb_name: str) -> Optional[str]:
    """Convert a full name to play-by-play format, e.g. 'Patrick Mahomes' -> 'P.Mahomes'.

    Drops suffixes ('Michael Penix Jr.' -> 'M.Penix') and keeps the last name's
    original capitalization ('J.J. McCarthy' -> 'J.McCarthy').
    """
    parts = qb_name.strip().split()
    while len(parts) > 2 and parts[-1].lower() in NAME_SUFFIXES:
        parts.pop()
    if len(parts) < 2:
        return None
    return f"{parts[0][0].upper()}.{parts[-1]}"


def normalize_name(name: str) -> str:
    """Lowercase and drop suffixes so 'Gardner Minshew II' matches 'Gardner Minshew'."""
    parts = str(name).strip().lower().split()
    while len(parts) > 2 and parts[-1] in NAME_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _name_eq(series: pd.Series, qb_name: str) -> pd.Series:
    """Case-insensitive, suffix-tolerant name match across datasets."""
    return series.astype(str).map(normalize_name) == normalize_name(qb_name)


def _season_ngs_row(ngs_qb: pd.DataFrame) -> pd.DataFrame:
    """NGS data includes weekly rows plus a season-total row (week 0). Prefer the total."""
    if "week" in ngs_qb.columns:
        season_total = ngs_qb[ngs_qb["week"] == 0]
        if not season_total.empty:
            return season_total
    return ngs_qb


# ---------------------------------------------------------------------------
# Stat extraction
# ---------------------------------------------------------------------------

def calculate_qb_epa_play(seasonal_qb: pd.DataFrame) -> float:
    """Average EPA per play (passes + rushes) over the season."""
    total_epa = seasonal_qb["rushing_epa"].values[0] + seasonal_qb["passing_epa"].values[0]
    total_plays = seasonal_qb["attempts"].values[0] + seasonal_qb["carries"].values[0]
    return safe_div(total_epa, total_plays)


def pressure_to_sack_percentage(weekly_qb: pd.DataFrame) -> float:
    """Percent of pressures that became sacks (regular season)."""
    return safe_div(weekly_qb["times_sacked"].sum(), weekly_qb["times_pressured"].sum()) * 100


def get_qb_stats(qb_name: str, data: dict[str, pd.DataFrame]) -> Optional[dict[str, Any]]:
    """Gather a QB's 2024 regular-season stats across all datasets.

    Returns a dict of plain Python values, or None if the player can't be
    found in any required dataset.
    """
    ngs, seasonal_pfr, weekly = data["ngs"], data["seasonal_pfr"], data["weekly"]
    pbp, ftn, seasonal = data["pbp"], data["ftn"], data["seasonal"]

    ngs_qb = ngs[_name_eq(ngs["player_display_name"], qb_name) & (ngs["season_type"] == "REG")]
    seasonal_pfr_qb = seasonal_pfr[_name_eq(seasonal_pfr["player"], qb_name)]
    seasonal_qb = seasonal[_name_eq(seasonal["name"], qb_name) & (seasonal["season_type"] == "REG")]
    weekly_qb = weekly[_name_eq(weekly["pfr_player_name"], qb_name) & (weekly["game_type"] == "REG")]

    pbp_name = format_name_pbp_passer(qb_name)
    if pbp_name is None:
        return None
    pbp_qb = pbp[_name_eq(pbp["passer_player_name"], pbp_name) & (pbp["season_type"] == "REG")]

    for frame in (ngs_qb, seasonal_pfr_qb, seasonal_qb, weekly_qb, pbp_qb):
        if frame.empty:
            return None

    ngs_qb = _season_ngs_row(ngs_qb)

    ftn_qb = ftn.merge(
        pbp_qb[["nflverse_game_id", "play_id"]],
        left_on=["nflverse_game_id", "nflverse_play_id"],
        right_on=["nflverse_game_id", "play_id"],
        how="inner",
    )

    attempts = ngs_qb["attempts"].values[0]
    turnover_worthy = ftn_qb["is_interception_worthy"].sum()

    stats = {
        "name": qb_name,
        "games_played": len(weekly_qb),
        "passer_rating": ngs_qb["passer_rating"].values[0],
        "pass_attempts": attempts,
        "pass_yards": ngs_qb["pass_yards"].values[0],
        "rush_yards": seasonal_qb["rushing_yards"].values[0],
        "rush_attempts": seasonal_qb["carries"].values[0],
        "pass_touchdowns": ngs_qb["pass_touchdowns"].values[0],
        "rush_touchdowns": seasonal_qb["rushing_tds"].values[0],
        "interceptions": ngs_qb["interceptions"].values[0],
        "completion_percentage": round(ngs_qb["completion_percentage"].values[0], 1),
        "completion_percentage_above_expectation": round(
            ngs_qb["completion_percentage_above_expectation"].values[0], 1
        ),
        "aggressiveness": round(ngs_qb["aggressiveness"].values[0], 1),
        "pressure_to_sack_percentage": round(pressure_to_sack_percentage(weekly_qb), 1),
        "epa_play": round(calculate_qb_epa_play(seasonal_qb), 2),
        "bad_throw_pct_average": round(seasonal_pfr_qb["bad_throw_pct"].values[0], 3),
        "sacks_taken": seasonal_qb["sacks"].values[0],
        "turnover_worthy_passes": turnover_worthy,
        "turnover_worthy_throw_rate": round(safe_div(turnover_worthy, attempts) * 100, 1),
        "avg_depth_of_target": round(seasonal_pfr_qb["intended_air_yards_per_pass_attempt"].values[0], 1),
        "yards_per_attempt": round(safe_div(ngs_qb["pass_yards"].values[0], attempts), 1),
        "avg_time_to_throw": round(ngs_qb["avg_time_to_throw"].values[0], 2),
        "fumbles_lost": seasonal_qb["rushing_fumbles_lost"].values[0]
        + seasonal_qb["sack_fumbles_lost"].values[0],
        "dakota": seasonal_qb["dakota"].values[0],
    }
    return {key: to_python(value) for key, value in stats.items()}


# ---------------------------------------------------------------------------
# Scoring (Adam Clark's weighted formula)
# ---------------------------------------------------------------------------

def generate_qb_score(stats: dict[str, Any]) -> float:
    """Score a QB's season with a weighted multi-metric formula. Higher is better."""
    total_yards = stats["pass_yards"] + stats["rush_yards"]
    # 250 total yards is roughly an average game in 2024
    yards_score = (safe_div(total_yards, stats["games_played"]) - 250) / 2

    # TD rate per pass/rush attempt, scaled by 200 because the raw rate is small
    total_tds = stats["pass_touchdowns"] + stats["rush_touchdowns"]
    total_plays = stats["pass_attempts"] + stats["rush_attempts"]
    touchdown_score = safe_div(total_tds, total_plays) * 200

    # Turnover-worthy throws per actual INT (luck factor), relative to bad-throw rate.
    # max(INTs, 1) avoids dividing by zero for QBs with no interceptions.
    luck_ratio = stats["turnover_worthy_passes"] / max(stats["interceptions"], 1)
    turnover_score = (
        safe_div(luck_ratio, stats["bad_throw_pct_average"])
        - stats["fumbles_lost"]
        - stats["interceptions"]
    )

    aggressiveness_score = (stats["avg_depth_of_target"] - 7.5) * stats["aggressiveness"]

    # Squared distance from a 2.5s baseline, keeping the sign
    ttt_delta = stats["avg_time_to_throw"] - 2.5
    if ttt_delta > 0:
        time_to_throw_score = (ttt_delta * 10) ** 2
    else:
        time_to_throw_score = -((ttt_delta * 10) ** 2)

    pressure_to_sack_score = safe_div(stats["pressure_to_sack_percentage"], stats["avg_time_to_throw"])

    score = (
        yards_score * 0.15
        + touchdown_score * 0.2
        + turnover_score * 0.15
        + aggressiveness_score * 0.05
        + (stats["epa_play"] * 30) * 0.25
        + time_to_throw_score * 0.05
        - pressure_to_sack_score * 0.15  # higher pressure-to-sack rate is worse
        - stats["sacks_taken"] * 0.1
        + stats["completion_percentage_above_expectation"] * 0.1
    )
    return round(score, 2)


def compare_qbs(stats_a: dict[str, Any], stats_b: dict[str, Any]) -> dict[str, Any]:
    """Score two QBs and report the winner. Used by the CLI and the AI agent tool."""
    score_a, score_b = generate_qb_score(stats_a), generate_qb_score(stats_b)
    if score_a == score_b:
        winner = None
    else:
        winner = stats_a["name"] if score_a > score_b else stats_b["name"]
    return {
        "scores": {stats_a["name"]: score_a, stats_b["name"]: score_b},
        "winner": winner,
        "margin": round(abs(score_a - score_b), 2),
    }
