# StatKit — Statistical Engineering & Simulation

I built this project as part of a statistical engineering assessment. The goal was to write a working statistics library and a probability simulation completely from scratch — no NumPy, no pandas, nothing external. Just Python and math.

It turned out to be a genuinely interesting challenge because you quickly realize how much work those libraries are quietly doing for you when you have to implement everything yourself.

---

## What's in here

The project has two main pieces.

The first is `DataAnalyzer` — a class that takes a list of numbers and computes all the standard descriptive statistics from scratch: mean, median, mode, variance (both sample and population), standard deviation, and outlier detection using z-scores. I also put a lot of effort into the error handling so that messy input — like None values, strings, booleans, or empty lists — fails clearly with a useful message rather than silently breaking.

The second is a Monte Carlo server crash simulation. The setup is: a startup server has a 4.5% chance of crashing on any given day. I simulate that across different time windows (30 days, 1 year, 1,000 days, 10,000 days) and track how close the observed crash rate gets to the theoretical 4.5% as the sample size grows. It's a clean way to watch the Law of Large Numbers happen in real time.

The salary dataset (`data/sample_salaries.json`) has 50 intentionally skewed salaries ranging from $38k up to $11.5M to show why the arithmetic mean alone can be a dangerously misleading number.

---

## Project structure

```
statkit/
│
├── data/
│   └── sample_salaries.json       # 50 startup salaries, heavily skewed toward executives
│
├── src/
│   ├── __init__.py                # Package init — imports DataAnalyzer and simulation tools
│   ├── stat_engine.py             # DataAnalyzer class — all stats written from scratch
│   └── monte_carlo.py             # Server crash simulation and LLN report
│
├── tests/
│   ├── __init__.py                # Marks tests/ as a package
│   └── test_stat_engine.py        # 56 unit tests covering all methods and edge cases
│
├── README.md
└── main.py                        # Run this to see the full analysis
```

---

## The math behind it

### Variance and why sample vs. population matters

There are two versions of variance and I implemented both.

**Population variance** is what you use when your data represents the entire group you care about. You divide by N:

```
σ² = Σ(xᵢ − μ)² / N
```

**Sample variance** is what you use when your data is just a sample drawn from a bigger population. The problem is that your sample mean is computed from the same data points, which causes the divide-by-N formula to systematically underestimate the true spread. The fix is Bessel's correction — you divide by N−1 instead:

```
s² = Σ(xᵢ − x̄)² / (N − 1)
```

Standard deviation is just the square root of variance in both cases. The reason we bother taking the square root is to get the result back into the same unit as the original data — so if the data is salaries in dollars, the standard deviation is also in dollars.

### How I handle the median for even vs. odd datasets

I sort the data first, then check whether the length is odd or even:

- **Odd length** → there's one element sitting exactly in the middle. I return `data[n // 2]`.
- **Even length** → there are two elements equally near the centre. I average them: `(data[n//2 − 1] + data[n//2]) / 2`.

For example, `[1, 2, 3, 4, 5]` has a median of 3. `[1, 2, 3, 4]` has a median of 2.5.

### Outlier detection with z-scores

For each data point I calculate how many standard deviations it sits from the mean:

```
z = (xᵢ − x̄) / s
```

If the absolute z-score exceeds the threshold (I default to 2.0), the point is flagged as an outlier. Setting the threshold at 2 captures roughly 95% of normally distributed data within the boundary, so only the genuinely extreme tail values get flagged.

One edge case I handled: when all values are identical the standard deviation is 0, which would cause a division by zero in the z-score formula. I catch that and return an empty list — no outliers possible when there's no spread at all.

### The Law of Large Numbers in the simulation

The LLN states that as the number of independent trials increases, the observed frequency of an event converges to its true theoretical probability. In the simulation, I run Bernoulli trials (flip a weighted coin with p = 0.045 each day) across four different time windows and track how close the observed crash rate gets to 4.5%:

| Days | Typical result | Distance from truth |
|------|---------------|---------------------|
| 30 | anywhere from 0% to 10% | high — mostly noise |
| 365 | closer, but still variable | moderate |
| 1,000 | noticeably tighter | low |
| 10,000 | very close to 4.5% | very low |

The key insight for the budget planning question: a startup that only has 30 days of crash data is essentially working with a coin flip. They might see zero crashes and budget nothing for maintenance, or they might see three crashes and massively overbuild their contingency fund. Neither observation is statistically trustworthy at that sample size.

---

## Getting started

You only need Python 3.8 or higher. There are no packages to install.

```bash
# 1. Clone the repo
git clone https://github.com/kidist-belay/statkit.git
cd statkit

# 2. Run the full analysis
python main.py
```

No pip install, no virtual environment, no setup.py. Just run it.

---

## Running the tests

```bash
# Run all tests with verbose output
python -m unittest discover -s tests -v

# Or run the test file directly
python tests/test_stat_engine.py
```

There are 56 tests covering all the main logic and edge cases. They should all pass on any Python 3.8+ installation.

---

## What the tests cover

I tried to test not just the happy path but the places where things could quietly go wrong:

- **Input validation** — strings, None, booleans, empty lists, wrong container types
- **Mean** — basics, negatives, floats, 100-element range
- **Median** — odd length, even length, unsorted input, repeated values, negatives
- **Mode** — single mode, bimodal, trimodal, all-unique message
- **Variance** — known population result (4.0), Bessel's correction (32/7), s² always > σ² for n > 1, single element edge cases
- **Standard deviation** — known value (2.0), relationship to variance verified to 12 decimal places
- **Outliers** — symmetric outliers detected, no false positives, bad threshold raises, zero std dev handled gracefully
- **Monte Carlo** — return structure, value ranges, LLN convergence at n=100k, seed reproducibility

---

## Acceptance criteria

- [x] Empty list raises `DatasetEmptyError` with a clear message
- [x] Mixed types raise `NonNumericDataError` and name the bad values
- [x] Booleans are explicitly rejected — they're secretly ints in Python and would corrupt the data silently
- [x] Multimodal datasets return all modes in a sorted list
- [x] All-unique datasets return a descriptive string rather than an empty list
- [x] Sample variance uses Bessel's correction (÷ N−1), verified against the known result 32/7
- [x] Population variance divides by N, verified against the known result 4.0
- [x] Sample variance is always greater than population variance for n > 1
- [x] Median handles both odd and even length datasets correctly
- [x] Standard deviation equals the square root of variance, verified to 12 decimal places
- [x] Outlier threshold of 0 or below raises `ValueError`
- [x] When standard deviation is 0, outlier detection returns an empty list without crashing
- [x] Monte Carlo simulation converges to within 0.5% of 4.5% at n = 100,000
- [x] Same random seed always produces the same simulation result
- [x] Zero external dependencies — only `math`, `random`, `typing`, `unittest`, `json`, `os`, `sys`

---

## Author

**Kidist Belay**

I wrote every formula from first principles for this assessment. The constraint of not using statistical libraries was actually useful — it forced me to understand *why* Bessel's correction exists and *what* a z-score is actually measuring, rather than just calling a function and trusting the output.
