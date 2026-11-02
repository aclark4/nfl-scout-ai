"""
build_stats.py - Precompute stats and scores for every qualifying QB.

Downloads the nflverse data once, runs the scoring formula for each QB, and
writes a small JSON file. The API and AI agent read that file instead of
downloading large play-by-play data on every startup.

Usage:
    python build_stats.py                       # 2024, QBs with 100+ attempts
    python build_stats.py --min-attempts 200 --out data/qb_stats_2024.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from qb_stats import generate_qb_score, get_qb_stats, load_data


def qualifying_qbs(data, min_attempts: int) -> list[str]:
    """QB names with at least `min_attempts` regular-season pass attempts (NGS season totals)."""
    ngs = data["ngs"]
    ngs = ngs[ngs["season_type"] == "REG"]
    if "week" in ngs.columns and (ngs["week"] == 0).any():
        ngs = ngs[ngs["week"] == 0]
        totals = ngs.groupby("player_display_name")["attempts"].max()
    else:
        totals = ngs.groupby("player_display_name")["attempts"].sum()
    return sorted(totals[totals >= min_attempts].index.tolist())


def build(year: int, min_attempts: int, out_path: Path) -> dict:
    data = load_data(year)
    names = qualifying_qbs(data, min_attempts)
    print(f"Found {len(names)} QBs with {min_attempts}+ attempts. Computing stats...")

    players, skipped = [], []
    for name in names:
        try:
            stats = get_qb_stats(name, data)
        except Exception as exc:  # keep going if one player has odd data
            skipped.append({"name": name, "reason": f"error: {exc}"})
            continue
        if stats is None:
            skipped.append({"name": name, "reason": "not found in every dataset"})
            continue
        stats["score"] = generate_qb_score(stats)
        players.append(stats)

    players.sort(key=lambda p: p["score"], reverse=True)
    for rank, player in enumerate(players, start=1):
        player["rank"] = rank

    result = {
        "season": year,
        "season_type": "REG",
        "min_attempts": min_attempts,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "player_count": len(players),
        "players": players,
        "skipped": skipped,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {len(players)} QBs to {out_path} ({len(skipped)} skipped)")
    for item in skipped:
        print(f"  skipped {item['name']}: {item['reason']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute QB stats and scores.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--min-attempts", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("data/qb_stats_2024.json"))
    args = parser.parse_args()
    build(args.year, args.min_attempts, args.out)


if __name__ == "__main__":
    main()
