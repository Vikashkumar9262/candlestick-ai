
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import (
    FEATURES,
    CLASSES,
    LABEL_TO_ID,
    make_features,
    make_labels,
    make_sequences,
)
from src.model import CandlestickCNN


DATA_PATH = Path("data/raw/nifty50.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 5
N_BOOTSTRAP = 2000
BOOTSTRAP_BLOCK_LENGTH = 5
RANDOM_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def moving_block_bootstrap(
    results: pd.DataFrame,
    patterns: list[str],
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Estimate 95% percentile confidence intervals using a circular
    moving-block bootstrap on chronologically ordered test observations.

    Returns are decimal returns, not percentages.
    """
    if results.empty:
        raise ValueError("Cannot bootstrap an empty results DataFrame.")

    if block_length < 1 or block_length > len(results):
        raise ValueError("Invalid bootstrap block length.")

    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replications.")

    returns = results["forward_return"].to_numpy(dtype=float)
    predicted = results["predicted_pattern"].to_numpy()
    n = len(results)

    if not np.isfinite(returns).all():
        raise ValueError("Forward returns contain non-finite values.")

    rng = np.random.default_rng(seed)
    offsets = np.arange(block_length)
    blocks_per_sample = int(np.ceil(n / block_length))

    # Bootstrap estimates for each predicted pattern.
    estimates = {
        pattern: {
            "mean": [],
            "excess": [],
            "difference_vs_others": [],
        }
        for pattern in patterns
    }

    baseline_estimates = []

    for _ in range(n_bootstrap):
        # Sample consecutive blocks to preserve some local dependence.
        starts = rng.integers(0, n, size=blocks_per_sample)

        sampled_idx = (
            starts[:, None] + offsets[None, :]
        ) % n

        sampled_idx = sampled_idx.ravel()[:n]

        boot_returns = returns[sampled_idx]
        boot_patterns = predicted[sampled_idx]

        baseline = boot_returns.mean()
        baseline_estimates.append(baseline)

        for pattern in patterns:
            mask = boot_patterns == pattern

            if not mask.any():
                estimates[pattern]["mean"].append(np.nan)
                estimates[pattern]["excess"].append(np.nan)
                estimates[pattern]["difference_vs_others"].append(
                    np.nan
                )
                continue

            group_mean = boot_returns[mask].mean()

            estimates[pattern]["mean"].append(group_mean)

            # Pattern group minus the overall test-sample mean.
            estimates[pattern]["excess"].append(
                group_mean - baseline
            )

            # Pattern group minus all other observations.
            other_mask = ~mask

            if other_mask.any():
                other_mean = boot_returns[other_mask].mean()

                estimates[pattern]["difference_vs_others"].append(
                    group_mean - other_mean
                )
            else:
                estimates[pattern]["difference_vs_others"].append(
                    np.nan
                )

    def confidence_interval(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return np.nan, np.nan

        lower, upper = np.percentile(values, [2.5, 97.5])
        return float(lower), float(upper)

    baseline_ci_low, baseline_ci_high = confidence_interval(
        baseline_estimates
    )

    rows = []

    for pattern in patterns:
        group = results.loc[
            results["predicted_pattern"] == pattern,
            "forward_return",
        ]

        if group.empty:
            continue

        other_group = results.loc[
            results["predicted_pattern"] != pattern,
            "forward_return",
        ]

        mean_ci_low, mean_ci_high = confidence_interval(
            estimates[pattern]["mean"]
        )

        excess_ci_low, excess_ci_high = confidence_interval(
            estimates[pattern]["excess"]
        )

        difference_ci_low, difference_ci_high = confidence_interval(
            estimates[pattern]["difference_vs_others"]
        )

        rows.append({
            "pattern": pattern,
            "samples": len(group),
            "mean_return": group.mean(),
            "mean_ci_lower": mean_ci_low,
            "mean_ci_upper": mean_ci_high,
            "baseline_return": returns.mean(),
            "baseline_ci_lower": baseline_ci_low,
            "baseline_ci_upper": baseline_ci_high,
            "excess_vs_baseline": group.mean() - returns.mean(),
            "excess_ci_lower": excess_ci_low,
            "excess_ci_upper": excess_ci_high,
            "difference_vs_others": (
                group.mean() - other_group.mean()
                if not other_group.empty
                else np.nan
            ),
            "difference_others_ci_lower": difference_ci_low,
            "difference_others_ci_upper": difference_ci_high,
        })

    return pd.DataFrame(rows).set_index("pattern")


def main():
    # --------------------------------------------------
    # 1. Load and prepare market data
    # --------------------------------------------------
    df = pd.read_csv(
        DATA_PATH,
        index_col="Date",
        parse_dates=True,
    )

    df = df.sort_index()

    features = make_features(df)

    if "Volume" not in df.columns:
        features["relative_volume"] = 1.0
    else:
        features["relative_volume"] = (
            features["relative_volume"].fillna(1.0)
        )

    features = (
        features.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    labels = make_labels(df).map(LABEL_TO_ID).to_numpy()

    n = len(df)
    val_end = int(n * 0.85)

    # --------------------------------------------------
    # 2. Load scaler, metadata, and trained model
    # --------------------------------------------------
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    metadata = joblib.load(MODEL_DIR / "metadata.joblib")

    window = metadata["window"]
    scaled = scaler.transform(features[FEATURES])

    X, y, idx = make_sequences(scaled, labels, window)

    # The final 15% is the held-out test period.
    test_mask = idx >= val_end

    X_test = X[test_mask]
    y_test = y[test_mask]
    test_idx = idx[test_mask]

    if len(X_test) == 0:
        raise ValueError(
            "No test sequences found. Check data and split."
        )

    model = CandlestickCNN(
        num_features=len(FEATURES),
        num_classes=len(CLASSES),
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_DIR / "candlestick_cnn.pt",
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    # --------------------------------------------------
    # 3. Predict candlestick patterns
    # --------------------------------------------------
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32)
        ),
        batch_size=128,
        shuffle=False,
    )

    predictions = []
    probabilities = []

    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=1)

            probabilities.extend(probs.cpu().numpy())
            predictions.extend(probs.argmax(dim=1).cpu().numpy())

    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    print("\nPattern recognition on held-out test data")

    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(CLASSES)),
            target_names=CLASSES,
            zero_division=0,
        )
    )

    print(
        "Balanced accuracy:",
        balanced_accuracy_score(y_test, predictions),
    )

    print(
        "Confusion matrix:\n",
        confusion_matrix(
            y_test,
            predictions,
            labels=np.arange(len(CLASSES)),
        ),
    )

    # --------------------------------------------------
    # 4. Calculate forward returns
    # --------------------------------------------------
    # Signal: candle i has closed.
    # Entry: open of candle i+1.
    # Exit: close of candle i+HORIZON.
    close = df["Close"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)

    records = []

    for j, i in enumerate(test_idx):
        exit_idx = i + HORIZON

        if exit_idx >= len(df):
            continue

        entry = open_[i + 1]
        exit_price = close[exit_idx]

        if (
            not np.isfinite(entry)
            or not np.isfinite(exit_price)
            or entry <= 0
        ):
            continue

        forward_return = exit_price / entry - 1.0

        records.append({
            "date": df.index[i],
            "actual_pattern": CLASSES[y_test[j]],
            "predicted_pattern": CLASSES[predictions[j]],
            "confidence": float(probabilities[j].max()),
            "forward_return": forward_return,
        })

    results = pd.DataFrame(records)

    if results.empty:
        raise ValueError("No valid test outcomes were generated.")

    # Preserve chronological order for block resampling.
    results = results.sort_values("date").reset_index(drop=True)

    predictions_path = REPORT_DIR / "test_predictions.csv"
    results.to_csv(predictions_path, index=False)

    # --------------------------------------------------
    # 5. Summarize outcomes by predicted pattern
    # --------------------------------------------------
    print(f"\nForward outcomes: {HORIZON}-candle horizon")

    baseline_return = results["forward_return"].mean()

    summary = results.groupby("predicted_pattern").agg(
        samples=("forward_return", "size"),
        mean_return=("forward_return", "mean"),
        median_return=("forward_return", "median"),
        positive_return_rate=(
            "forward_return",
            lambda s: (s > 0).mean(),
        ),
        return_std=("forward_return", "std"),
    )

    summary["excess_vs_baseline"] = (
        summary["mean_return"] - baseline_return
    )

    print("\nPattern groups versus unconditional baseline")
    print(summary.round(5).to_string())

    print(f"\nOverall mean forward return: {baseline_return:.6f}")
    print(
        "Overall mean forward return (%): "
        f"{baseline_return * 100:.3f}%"
    )

    summary.to_csv(REPORT_DIR / "pattern_outcomes.csv")
    summary.to_csv(
        REPORT_DIR / "pattern_baseline_comparison.csv"
    )

    # --------------------------------------------------
    # 6. Statistical reliability: moving-block bootstrap
    # --------------------------------------------------
    print("\nRunning moving-block bootstrap...")

    statistical_results = moving_block_bootstrap(
        results=results,
        patterns=CLASSES,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        n_bootstrap=N_BOOTSTRAP,
        seed=RANDOM_SEED,
    )

    statistical_path = REPORT_DIR / "pattern_statistical_tests.csv"
    statistical_results.to_csv(statistical_path)

    display_columns = [
        "samples",
        "mean_return",
        "mean_ci_lower",
        "mean_ci_upper",
        "excess_vs_baseline",
        "excess_ci_lower",
        "excess_ci_upper",
        "difference_vs_others",
        "difference_others_ci_lower",
        "difference_others_ci_upper",
    ]

    print("\nBootstrap 95% confidence intervals")
    print(
        statistical_results[display_columns]
        .round(5)
        .to_string()
    )

    print("\nSaved reports:")
    print(f"- {predictions_path}")
    print(f"- {REPORT_DIR / 'pattern_outcomes.csv'}")
    print(f"- {REPORT_DIR / 'pattern_baseline_comparison.csv'}")
    print(f"- {statistical_path}")


if __name__ == "__main__":
    main()
