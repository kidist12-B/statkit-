"""
stat_engine.py
==============
Core statistical analysis module — built from scratch using only
Python's standard library.

The main class is DataAnalyzer. It accepts a raw list or tuple of
numbers, cleans the input, and exposes methods for computing
descriptive statistics and detecting outliers.

No third-party libraries are used anywhere in this file.
Standard library only: math, typing.
"""

import math
from typing import Union


# ---------------------------------------------------------------------------
# Custom exception classes
# ---------------------------------------------------------------------------

class DatasetEmptyError(ValueError):
    """
    Raised when DataAnalyzer receives an empty dataset or when a
    statistical operation becomes undefined due to insufficient data.
    """
    def __str__(self):
        return (
            "DatasetEmptyError: Cannot perform analysis on an empty dataset. "
            "Please pass at least one valid numeric value."
        )


class NonNumericDataError(TypeError):
    """
    Raised when one or more values in the input cannot be interpreted
    as a number. The offending values are stored and displayed.
    """
    def __init__(self, rejected: list):
        self.rejected = rejected

    def __str__(self):
        return (
            f"NonNumericDataError: The following values are not numeric and "
            f"cannot be processed: {self.rejected}. "
            "Only int and float values are accepted."
        )


# ---------------------------------------------------------------------------
# DataAnalyzer
# ---------------------------------------------------------------------------

class DataAnalyzer:
    """
    Descriptive statistics engine for 1-D numerical datasets.

    Parameters
    ----------
    raw_data : list | tuple
        The input dataset. Must contain only int or float values.
        Booleans, strings, None, and other non-numeric types will
        trigger NonNumericDataError.

    Attributes
    ----------
    values : list[float]
        Cleaned and validated dataset stored as floats.
    count  : int
        Total number of data points after cleaning.

    Raises
    ------
    DatasetEmptyError     – if cleaned dataset has zero elements
    NonNumericDataError   – if any value fails numeric validation
    TypeError             – if raw_data is not a list or tuple
    """

    def __init__(self, raw_data: Union[list, tuple]) -> None:
        if not isinstance(raw_data, (list, tuple)):
            raise TypeError(
                f"DataAnalyzer expects a list or tuple, got {type(raw_data).__name__}."
            )

        valid, rejected = self._validate(raw_data)

        if rejected:
            raise NonNumericDataError(rejected)

        if not valid:
            raise DatasetEmptyError()

        self.values: list = valid
        self.count: int = len(self.values)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(raw: Union[list, tuple]):
        """
        Walk through raw input and separate numeric from non-numeric values.

        Booleans are explicitly blocked — in Python, bool is a subclass of
        int, so True and False would silently pass a plain isinstance(x, int)
        check. We catch and reject them first to avoid corrupting the dataset.

        Returns (valid_floats, rejected_originals).
        """
        valid = []
        rejected = []

        for item in raw:
            if isinstance(item, bool):
                rejected.append(item)
            elif isinstance(item, (int, float)):
                valid.append(float(item))
            else:
                rejected.append(item)

        return valid, rejected

    # ------------------------------------------------------------------
    # Central tendency
    # ------------------------------------------------------------------

    def get_mean(self) -> float:
        """
        Arithmetic mean.

        Formula:  x̄ = (x₁ + x₂ + ... + xₙ) / n

        Walks through every value, sums them, divides by the count.
        Simple but can be misleading when the dataset is heavily skewed.
        """
        total = 0.0
        for v in self.values:
            total += v
        return total / self.count

    def get_median(self) -> float:
        """
        Middle value of the sorted dataset.

        Steps:
          1. Sort a copy of the data (never mutate the original).
          2. If n is odd  → return the single centre element.
          3. If n is even → return the average of the two centre elements.

        Unlike the mean, the median is resistant to extreme outliers,
        making it a better measure of the 'typical' value for skewed data.
        """
        ordered = sorted(self.values)
        mid = self.count // 2

        if self.count % 2 != 0:
            # Odd length — one clear middle element
            return ordered[mid]
        else:
            # Even length — average the two neighbours of the midpoint
            return (ordered[mid - 1] + ordered[mid]) / 2.0

    def get_mode(self) -> Union[list, str]:
        """
        Most frequently occurring value(s).

        Build a frequency table manually (no collections.Counter),
        then find the highest frequency and collect all values that
        share it.

        Returns
        -------
        list[float]
            All modes sorted in ascending order, e.g. [3.0, 7.0].
        str
            "No mode — every value appears exactly once."
            when the dataset has no repeated values.
        """
        frequency: dict = {}
        for v in self.values:
            frequency[v] = frequency.get(v, 0) + 1

        peak = max(frequency.values())

        if peak == 1:
            return "No mode — every value appears exactly once."

        return sorted(k for k, freq in frequency.items() if freq == peak)

    # ------------------------------------------------------------------
    # Dispersion
    # ------------------------------------------------------------------

    def get_variance(self, population: bool = False) -> float:
        """
        Average squared distance from the mean.

        Two modes:
          Population (σ²) — divide by N.
              Use when your dataset IS the entire population.
              σ² = Σ(xᵢ − μ)² / N

          Sample (s²) — divide by N−1 (Bessel's correction).
              Use when your dataset is a sample from a larger population.
              s² = Σ(xᵢ − x̄)² / (N − 1)

        Why N−1?
          The sample mean x̄ is calculated from the same data points, which
          causes the ÷N formula to slightly underestimate the true spread.
          Subtracting 1 from the denominator corrects this bias and produces
          an unbiased estimator of the population variance.

        Parameters
        ----------
        population : bool, default False
            False → sample variance  (÷ N−1)
            True  → population variance (÷ N)

        Raises
        ------
        DatasetEmptyError
            When population=False and count == 1, because N−1 = 0.
        """
        if not population and self.count == 1:
            raise DatasetEmptyError()

        mean = self.get_mean()
        squared_diffs = [(x - mean) ** 2 for x in self.values]
        divisor = self.count if population else (self.count - 1)
        return sum(squared_diffs) / divisor

    def get_standard_deviation(self, population: bool = False) -> float:
        """
        Square root of variance.

        Brings the unit back to the same scale as the original data,
        making it directly interpretable (e.g. '± $X salary').

        Parameters
        ----------
        population : bool, default False
            Passed through to get_variance().
        """
        return math.sqrt(self.get_variance(population=population))

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def get_outliers(self, threshold: float = 2.0) -> list:
        """
        Identify data points that sit unusually far from the mean.

        Method: z-score
            z = (xᵢ − x̄) / s

        Any point with |z| > threshold is considered an outlier.
        The default threshold of 2.0 flags roughly the outermost 5%
        of a normal distribution.

        Parameters
        ----------
        threshold : float, default 2.0
            Must be a positive number.

        Returns
        -------
        list[float]
            Sorted list of outlier values. Empty list if none found.

        Raises
        ------
        ValueError
            If threshold is zero or negative.
        """
        if threshold <= 0:
            raise ValueError(
                f"threshold must be positive, got {threshold}."
            )

        std = self.get_standard_deviation()

        # When all values are identical the std dev is 0 — no outliers possible
        if std == 0:
            return []

        mean = self.get_mean()
        result = []

        for x in self.values:
            z_score = abs(x - mean) / std
            if z_score > threshold:
                result.append(x)

        return sorted(result)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def full_report(self) -> dict:
        """Return a dictionary containing all computed statistics."""
        return {
            "count"               : self.count,
            "mean"                : self.get_mean(),
            "median"              : self.get_median(),
            "mode"                : self.get_mode(),
            "sample_variance"     : self.get_variance(population=False),
            "population_variance" : self.get_variance(population=True),
            "sample_std"          : self.get_standard_deviation(population=False),
            "population_std"      : self.get_standard_deviation(population=True),
            "outliers_at_2std"    : self.get_outliers(threshold=2.0),
        }

    def __repr__(self):
        sample = self.values[:4]
        tail = f"... +{self.count - 4} more" if self.count > 4 else ""
        return f"DataAnalyzer(count={self.count}, preview={sample}{tail})"
