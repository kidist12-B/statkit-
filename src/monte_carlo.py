"""
monte_carlo.py
==============
Server reliability simulation using the Monte Carlo method.

Scenario:
    A startup's production server has a 4.5% (0.045) theoretical
    probability of crashing on any given day.

This module simulates that daily risk across varying time windows
to demonstrate the Law of Large Numbers (LLN):

    As the number of trials grows toward infinity, the observed
    (empirical) frequency of an event converges to its true
    theoretical probability.

Standard library only: random.
"""

import random

# The ground-truth crash probability — this is what the LLN converges toward
CRASH_PROBABILITY: float = 0.045


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def crash_trial(rng: random.Random) -> bool:
    """
    Simulate one day of server operation.

    A single Bernoulli trial with p = CRASH_PROBABILITY.
    Draws a uniform random float in [0, 1).
    Returns True (server crashed) if the draw falls below the threshold.

    Parameters
    ----------
    rng : random.Random
        An isolated Random instance so callers control reproducibility
        without touching the global random state.
    """
    return rng.random() < CRASH_PROBABILITY


# ---------------------------------------------------------------------------
# Multi-day simulation
# ---------------------------------------------------------------------------

def run_server_simulation(days: int, seed: int = None):
    """
    Simulate *days* of server operation and return the results.

    Parameters
    ----------
    days : int
        Number of days to simulate. Must be at least 1.
    seed : int, optional
        Seed for the random number generator. Provide a fixed value
        to get reproducible results.

    Returns
    -------
    dict with keys:
        days            – the input trial length
        total_crashes   – how many crash events occurred
        observed_prob   – total_crashes / days
        theoretical     – the known true probability (0.045)
        error           – absolute difference between observed and theoretical

    Raises
    ------
    ValueError
        If days is less than 1.
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, received {days}.")

    rng = random.Random(seed)
    crashes = sum(1 for _ in range(days) if crash_trial(rng))
    observed = crashes / days

    return {
        "days"          : days,
        "total_crashes" : crashes,
        "observed_prob" : observed,
        "theoretical"   : CRASH_PROBABILITY,
        "error"         : abs(observed - CRASH_PROBABILITY),
    }


# ---------------------------------------------------------------------------
# LLN demonstration — run multiple trial sizes and print a report
# ---------------------------------------------------------------------------

def lln_report(trial_sizes: list = None, seed: int = 42) -> list:
    """
    Run the simulation at several different trial sizes and print
    a formatted table showing how the observed probability converges
    toward the theoretical value as n increases.

    Parameters
    ----------
    trial_sizes : list[int], optional
        List of day counts to test. Defaults to [30, 365, 1000, 10000].
    seed : int
        Base seed. Each trial uses seed + index so they are independent
        but still reproducible.

    Returns
    -------
    list[dict]
        One result dictionary per trial size.
    """
    if trial_sizes is None:
        trial_sizes = [30, 365, 1_000, 10_000]

    results = []
    divider = "─" * 68

    print(f"\n{'═'*68}")
    print("  SERVER CRASH SIMULATION  —  Law of Large Numbers")
    print(f"  Theoretical crash probability: {CRASH_PROBABILITY:.1%} per day")
    print(f"{'═'*68}")
    print(f"  {'Days':>9}  {'Crashes':>8}  {'Observed':>10}  {'Theory':>8}  {'|Error|':>10}")
    print(f"  {divider}")

    for i, days in enumerate(trial_sizes):
        result = run_server_simulation(days, seed=seed + i)
        results.append(result)

        print(
            f"  {days:>9,}  "
            f"{result['total_crashes']:>8,}  "
            f"{result['observed_prob']:>10.4%}  "
            f"{result['theoretical']:>8.4%}  "
            f"{result['error']:>10.4%}"
        )

    print(f"  {divider}")
    _print_lln_commentary(results)
    return results


def _print_lln_commentary(results: list) -> None:
    """Print a plain-language explanation of what the numbers show."""
    first = results[0]
    last  = results[-1]

    print(f"\n  WHAT THIS TELLS US\n  {'─'*66}\n")
    print(
        f"  At {first['days']} days the observed rate was {first['observed_prob']:.2%}.\n"
        f"  That is {first['error']:.2%} away from the true rate of {CRASH_PROBABILITY:.2%}.\n"
        f"  With so few data points the result is basically noise —\n"
        f"  you might see 0 crashes (0%) or 3 crashes (10%), both by chance.\n"
    )
    print(
        f"  At {last['days']:,} days the observed rate was {last['observed_prob']:.2%}.\n"
        f"  The error has shrunk to just {last['error']:.2%}.\n"
        f"  This is the Law of Large Numbers: more trials → less variance\n"
        f"  → the simulated result homes in on the true probability.\n"
    )
    print(
        f"  WHY 30 DAYS IS DANGEROUS FOR BUDGET PLANNING\n"
        f"  {'─'*66}\n"
        f"  If a startup only looks at one month of server logs before\n"
        f"  setting their annual maintenance budget, they are essentially\n"
        f"  guessing. The 30-day sample lives in a high-variance zone where\n"
        f"  random luck dominates. A run of zero crashes feels like good\n"
        f"  news but is statistically meaningless at that sample size.\n"
        f"  The LLN only kicks in reliably after hundreds of observations.\n"
        f"  Smart budgeting requires either long historical data or a\n"
        f"  formal probability model — not a single month of evidence.\n"
    )
