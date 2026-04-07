"""
main.py
=======
Entry point for the StatKit project.

Run with:
    python main.py

Executes two analyses back-to-back:
  1. Full descriptive statistics on the startup salary dataset,
     including outlier detection and a commentary on why mean
     alone is not a reliable metric for skewed data.

  2. Monte Carlo server crash simulation across four time windows
     to illustrate the Law of Large Numbers.
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.stat_engine import DataAnalyzer
from src.monte_carlo import lln_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def currency(value) -> str:
    if isinstance(value, float):
        return f"${value:>15,.2f}"
    return str(value)

def section(title: str) -> None:
    print(f"\n{'═'*68}")
    print(f"  {title}")
    print(f"{'═'*68}")


# ---------------------------------------------------------------------------
# Part 1 — Salary analysis
# ---------------------------------------------------------------------------

def salary_analysis() -> None:
    section("PART 1 — Startup Salary Dataset  |  Descriptive Statistics")

    path = os.path.join(BASE_DIR, "data", "sample_salaries.json")
    with open(path) as f:
        payload = json.load(f)

    salaries = payload["salaries"]
    print(f"\n  Loaded {len(salaries)} salary records.")
    print(f"  First 8 values: {salaries[:8]} …\n")

    da = DataAnalyzer(salaries)
    report = da.full_report()

    print(f"  {'─'*64}")
    print(f"  {'Metric':<28}  {'Value':>30}")
    print(f"  {'─'*64}")
    rows = [
        ("Count",                  report["count"]),
        ("Mean",                   currency(report["mean"])),
        ("Median",                 currency(report["median"])),
        ("Mode",                   report["mode"]),
        ("Sample Variance",        currency(report["sample_variance"])),
        ("Population Variance",    currency(report["population_variance"])),
        ("Sample Std Dev",         currency(report["sample_std"])),
        ("Population Std Dev",     currency(report["population_std"])),
    ]
    for label, val in rows:
        print(f"  {label:<28}  {str(val):>30}")
    print(f"  {'─'*64}")

    outliers2 = da.get_outliers(threshold=2.0)
    outliers3 = da.get_outliers(threshold=3.0)

    print(f"\n  Outliers beyond 2 standard deviations  [{len(outliers2)} found]")
    for o in outliers2:
        print(f"    {currency(o)}")

    print(f"\n  Outliers beyond 3 standard deviations  [{len(outliers3)} found]")
    for o in outliers3:
        print(f"    {currency(o)}")

    mean   = report["mean"]
    median = report["median"]
    std    = report["sample_std"]

    print(f"""
  {'─'*64}
  WHY THE MEAN IS MISLEADING HERE
  {'─'*64}

  Mean   : {currency(mean)}
  Median : {currency(median)}
  Std Dev: {currency(std)}

  The mean salary sits at {currency(mean).strip()} — a figure that most
  employees will never come close to earning. It is inflated
  by a tiny group of C-suite executives whose compensation
  packages dwarf everyone else's.

  The median of {currency(median).strip()} is a far more honest number.
  It tells you what the person sitting in the middle of the
  salary ladder actually earns. More than half the workforce
  earns below this figure.

  The standard deviation of {currency(std).strip()} shows just how
  spread out the data really is. A spread that wide relative
  to the mean is a textbook sign of a right-skewed distribution
  driven by extreme high-end outliers.

  Lesson: for skewed distributions, always report the median
  alongside the mean — and never ignore the standard deviation.
    """)


# ---------------------------------------------------------------------------
# Part 2 — Monte Carlo / LLN
# ---------------------------------------------------------------------------

def server_simulation() -> None:
    section("PART 2 — Monte Carlo Simulation  |  Law of Large Numbers")
    lln_report(trial_sizes=[30, 365, 1_000, 10_000], seed=42)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "★" * 68)
    print("  StatKit  —  Statistical Engineering & Simulation")
    print("  Author: Kidist Belay")
    print("★" * 68)

    salary_analysis()
    server_simulation()

    print("\n✅  Done.\n")

