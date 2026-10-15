
from pathlib import Path
import math

import numpy as np
import pandas as pd

REPORT_DIR = Path("reports")
INPUT_FILE = REPORT_DIR / "test_predictions.csv"

# Sensitivity analysis: number of consecutive test observations per block.
BLOCK_LENGTHS = [5, 10, 20]
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

# Five-candle forward returns overlap. A HAC lag of 20 is a
# conservative starting point, not a guarantee of independence.
HAC_LAG = 20


def moving_block_indices(n, block_length, rng):
    """Circular moving-block bootstrap indices."""
    if not 1 <= block_length <= n:
        raise ValueError(
            f"Block length must be between 1 and {n}."
        )

    blocks_needed = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=blocks_needed)
    offsets = np.arange(block_length)

    indices = (
        starts[:, None] + offsets[None, :]
    ) % n

    return indices.ravel()[:n]


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)

    adjusted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max

    return adjusted


def hac_group_test(returns, is_group, max_lag=HAC_LAG):
    """
    Estimate group-minus-other mean return difference using
    intercept + group-indicator OLS with Newey-West HAC SE.

    Returns (difference, standard_error, p_value).
    The p-value uses a normal approximation.
    """
    y = np.asarray(returns, dtype=float)
    indicator = np.asarray(is_group, dtype=float)

    x = np.column_stack([np.ones(len(y)), indicator])
    n, k = x.shape

    if indicator.sum() == 0 or indicator.sum() == n:
        return np.nan, np.nan, np.nan

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    residuals = y - x @ beta

    # Newey-West sandwich "meat".
    scores = x * residuals[:, None]
    meat = scores.T @ scores

    lag_limit = min(max_lag, n - 1)

    for lag in range(1, lag_limit + 1):
        weight = 1.0 - lag / (lag_limit + 1)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)

    covariance = xtx_inv @ meat @ xtx_inv
    variance = max(float(covariance[1, 1]), 0.0)
    se = math.sqrt(variance)

    difference = float(beta[1])

    if se == 0:
        p_value = 0.0 if difference != 0 else 1.0
    else:
        z = difference / se
        p_value = math.erfc(abs(z) / math.sqrt(2))

    return difference, se, p_value


def bootstrap_pattern_stats(
    returns,
    predicted,
    pattern,
    block_length,
    n_bootstrap,
    rng,
):
    """Bootstrap mean, excess versus overall baseline, and
    difference versus observations assigned to other patterns."""
    n = len(returns)
    original_mask = predicted == pattern

    if not original_mask.any() or original_mask.all():
        return None

    original_mean = returns[original_mask].mean()
    baseline_mean = returns.mean()
    other_mean = returns[~original_mask].mean()

    boot_mean = []
    boot_excess = []
    boot_difference = []

    for _ in range(n_bootstrap):
        idx = moving_block_indices(n, block_length, rng)
        y = returns[idx]
        labels = predicted[idx]

        mask = labels == pattern

        if not mask.any() or mask.all():
            continue

        group_mean = y[mask].mean()
        other_group_mean = y[~mask].mean()

        boot_mean.append(group_mean)
        boot_excess.append(group_mean - y.mean())
        boot_difference.append(group_mean - other_group_mean)

    def ci(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) < 100:
            return np.nan, np.nan

        return tuple(np.percentile(values, [2.5, 97.5]))

    mean_low, mean_high = ci(boot_mean)
    excess_low, excess_high = ci(boot_excess)
    diff_low, diff_high = ci(boot_difference)

    return {
        "pattern": pattern,
        "block_length": block_length,
        "samples": int(original_mask.sum()),
        "mean_return": original_mean,
        "mean_ci_lower": mean_low,
        "mean_ci_upper": mean_high,
        "baseline_return": baseline_mean,
        "excess_vs_baseline": original_mean - baseline_mean,
        "excess_ci_lower": excess_low,
        "excess_ci_upper": excess_high,
        "difference_vs_others": original_mean - other_mean,
        "difference_ci_lower": diff_low,
        "difference_ci_upper": diff_high,
        "valid_bootstrap_reps": len(boot_mean),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run `python -m src.evaluate` first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {"forward_return", "predicted_pattern"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {INPUT_FILE}: {sorted(missing)}"
        )

    # Preserve chronological row order. Do not shuffle before bootstrapping.
    df = df.dropna(subset=["forward_return", "predicted_pattern"]).copy()
    df["forward_return"] = pd.to_numeric(
        df["forward_return"], errors="coerce"
    )
    df = df.dropna(subset=["forward_return"]).reset_index(drop=True)

    returns = df["forward_return"].to_numpy(dtype=float)
    predicted = df["predicted_pattern"].astype(str).to_numpy()

    if len(returns) < max(BLOCK_LENGTHS) * 2:
        raise ValueError(
            "Too few observations for the requested block lengths."
        )

    if not np.isfinite(returns).all():
        raise ValueError("Forward returns contain non-finite values.")

    patterns = sorted(np.unique(predicted))
    rng = np.random.default_rng(RANDOM_SEED)

    # Bootstrap sensitivity across block lengths.
    bootstrap_rows = []

    for block_length in BLOCK_LENGTHS:
        print(f"Bootstrapping with block length {block_length}...")

        for pattern in patterns:
            result = bootstrap_pattern_stats(
                returns=returns,
                predicted=predicted,
                pattern=pattern,
                block_length=block_length,
                n_bootstrap=N_BOOTSTRAP,
                rng=rng,
            )

            if result is not None:
                bootstrap_rows.append(result)

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_path = REPORT_DIR / "robustness_by_block_length.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)

    # One HAC test per pattern: difference between that pattern and
    # all other predicted-pattern observations.
    test_rows = []

    for pattern in patterns:
        mask = predicted == pattern

        difference, se, p_value = hac_group_test(
            returns,
            mask,
            max_lag=HAC_LAG,
        )

        test_rows.append({
            "pattern": pattern,
            "samples": int(mask.sum()),
            "mean_return": float(returns[mask].mean()),
            "other_mean_return": (
                float(returns[~mask].mean()) if (~mask).any() else np.nan
            ),
            "difference_vs_others": difference,
            "hac_standard_error": se,
            "hac_lag": HAC_LAG,
            "p_value": p_value,
        })

    tests_df = pd.DataFrame(test_rows)
    tests_df["holm_adjusted_p"] = holm_adjust(tests_df["p_value"])
    tests_df["significant_holm_0_05"] = (
        tests_df["holm_adjusted_p"] < 0.05
    )

    tests_path = REPORT_DIR / "pattern_holm_tests.csv"
    tests_df.to_csv(tests_path, index=False)

    print("\n=== HAC tests with Holm correction ===")
    print(
        tests_df[
            [
                "pattern",
                "samples",
                "difference_vs_others",
                "hac_standard_error",
                "p_value",
                "holm_adjusted_p",
                "significant_holm_0_05",
            ]
        ].round(6).to_string(index=False)
    )

    print("\n=== Bootstrap sensitivity ===")
    columns = [
        "pattern",
        "block_length",
        "samples",
        "excess_vs_baseline",
        "excess_ci_lower",
        "excess_ci_upper",
        "difference_ci_lower",
        "difference_ci_upper",
    ]
    print(bootstrap_df[columns].round(6).to_string(index=False))

    print("\nSaved reports:")
    print(f"- {bootstrap_path}")
    print(f"- {tests_path}")


if __name__ == "__main__":
    main()
