"""
player_comparison.py - Interactive CLI that compares two QBs' 2024 seasons.

Same experience as the original tool, now built on the reusable qb_stats module.
"""

from qb_stats import compare_qbs, get_qb_stats, load_data

ROWS = [
    ("Passing Yards", "pass_yards"),
    ("Passing Touchdowns", "pass_touchdowns"),
    ("Interceptions", "interceptions"),
    ("Completion Percentage", "completion_percentage"),
    ("EPA/Play", "epa_play"),
    ("Rushing Yards", "rush_yards"),
    ("Rushing Touchdowns", "rush_touchdowns"),
    ("Turnover Worthy Throws", "turnover_worthy_passes"),
    ("Bad Throw Rate", "bad_throw_pct_average"),
    ("Average Depth of Target", "avg_depth_of_target"),
    ("Sacks", "sacks_taken"),
    ("Pressure to Sack Percentage", "pressure_to_sack_percentage"),
    ("Average Time to Throw", "avg_time_to_throw"),
]


def output_stats(qb1_stats: dict, qb2_stats: dict) -> None:
    print("QB Stat Comparison:")
    print(f"{'Name:':<35}{qb1_stats['name']:<25}{qb2_stats['name']}")
    for label, key in ROWS:
        print(f"{label + ':':<35}{str(qb1_stats[key]):<25}{qb2_stats[key]}")
    print()


def prompt_for_qb(prompt: str, data: dict, exclude: str = "") -> dict:
    while True:
        name = input(prompt).strip()
        if exclude and name.lower() == exclude.lower():
            print("You entered the same quarterback twice!")
            continue
        stats = get_qb_stats(name, data)
        if stats is None:
            print(f"{name} is not a valid player. Try again.")
            continue
        return stats


def main() -> None:
    data = load_data(2024)
    qb1_stats = prompt_for_qb("Enter the first Quarterback: ", data)
    qb2_stats = prompt_for_qb("Enter the second Quarterback: ", data, exclude=qb1_stats["name"])

    result = compare_qbs(qb1_stats, qb2_stats)
    qb1, qb2 = qb1_stats["name"], qb2_stats["name"]
    s1, s2 = result["scores"][qb1], result["scores"][qb2]

    if result["winner"] is None:
        print(f"WOW! They both got the same score of {s1}")
    else:
        loser = qb2 if result["winner"] == qb1 else qb1
        print(f"{result['winner']} scores higher than {loser} according to my algorithm!")
        print(f"{qb1} had a score of {s1}, and {qb2} had a score of {s2}")

    answer = input(f"Type 'y' to see a stat comparison between {qb1} and {qb2}: ").strip().lower()
    if answer == "y":
        output_stats(qb1_stats, qb2_stats)

    print("Thanks for using my 2024 QB Comparison Tool!")


if __name__ == "__main__":
    main()
