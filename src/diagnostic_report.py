from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)

REPORTS = Path("reports")
OUT = REPORTS / "diagnostics"
PRED_PATH = REPORTS / "final_holdout_predictions.csv"
TRADES_PATH = REPORTS / "final_holdout_trades.csv"
SUMMARY_PATH = REPORTS / "final_holdout_summary.json"
OUT.mkdir(parents=True, exist_ok=True)


def pct(value):
    return "N/A" if pd.isna(value) else f"{value:.3f}%"


def compound_pct(returns):
    values = pd.to_numeric(pd.Series(returns), errors="coerce").dropna().to_numpy()
    return float((np.prod(1 + values) - 1) * 100) if len(values) else np.nan


def pf(returns):
    r = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def main():
    for path in (PRED_PATH, TRADES_PATH, SUMMARY_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run this script from the candlestick-ai project root."
            )

    pred = pd.read_csv(PRED_PATH)
    trades = pd.read_csv(TRADES_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    required_pred = {"actual_pattern", "predicted_pattern"}
    required_trades = {
        "predicted_pattern", "confidence", "gross_return", "net_return", "exit_date"
    }
    if not required_pred.issubset(pred.columns):
        raise ValueError(f"Predictions CSV needs {required_pred}; found {list(pred.columns)}")
    if not required_trades.issubset(trades.columns):
        raise ValueError(f"Trades CSV needs {required_trades}; found {list(trades.columns)}")

    pred = pred.dropna(subset=["actual_pattern", "predicted_pattern"]).copy()
    pred["actual_pattern"] = pred["actual_pattern"].astype(str)
    pred["predicted_pattern"] = pred["predicted_pattern"].astype(str)
    known = ["none", "doji", "hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing"]
    present = set(pred["actual_pattern"]) | set(pred["predicted_pattern"])
    labels = [label for label in known if label in present]
    labels += sorted(present - set(labels))

    y_true, y_pred = pred["actual_pattern"], pred["predicted_pattern"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).rename_axis("actual_pattern").to_csv(
        OUT / "confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title("Final holdout confusion matrix")
    ax.set_xlabel("Predicted pattern")
    ax.set_ylabel("Actual pattern")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    cut = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cut else "black")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    class_df = pd.DataFrame(classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )).T
    class_df.to_csv(OUT / "classification_report.csv", index_label="class")

    trades["confidence"] = pd.to_numeric(trades["confidence"], errors="coerce")
    trades["gross_return"] = pd.to_numeric(trades["gross_return"], errors="coerce")
    trades["net_return"] = pd.to_numeric(trades["net_return"], errors="coerce")
    trades["exit_date"] = pd.to_datetime(trades["exit_date"], errors="coerce")
    trades = trades.dropna(subset=["confidence", "net_return", "exit_date"]).sort_values("exit_date").copy()

    def stats(group):
        r = group["net_return"]
        return {
            "trades": len(group),
            "win_rate_pct": (r.gt(0).mean() * 100) if len(group) else np.nan,
            "mean_net_return_pct": (r.mean() * 100) if len(group) else np.nan,
            "median_net_return_pct": (r.median() * 100) if len(group) else np.nan,
            "sum_net_returns_pct_noncompounded": r.sum() * 100,
            "compounded_net_return_pct": compound_pct(r),
            "profit_factor": pf(r),
        }

    by_pattern = pd.DataFrame([
        {"predicted_pattern": name, **stats(group)}
        for name, group in trades.groupby("predicted_pattern", dropna=False)
    ])
    by_pattern.to_csv(OUT / "trades_by_pattern.csv", index=False)

    # These bands describe the already-tested trades; do not use them to tune this holdout.
    trades["confidence_band"] = pd.cut(
        trades["confidence"], bins=[0.70, 0.80, 0.90, 1.0000001],
        labels=["0.70–<0.80", "0.80–<0.90", "0.90–1.00"],
        right=False, include_lowest=True
    )
    by_confidence = pd.DataFrame([
        {"confidence_band": str(name), **stats(group)}
        for name, group in trades.groupby("confidence_band", observed=False)
    ])
    by_confidence.to_csv(OUT / "trades_by_confidence.csv", index=False)

    trades["equity"] = (1 + trades["net_return"]).cumprod()
    trades["running_peak"] = trades["equity"].cummax()
    trades["drawdown_pct"] = (trades["equity"] / trades["running_peak"] - 1) * 100
    trades.to_csv(OUT / "trades_with_equity_and_drawdown.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(trades["exit_date"], trades["equity"])
    ax.set_title("Final holdout strategy equity (completed trades)")
    ax.set_xlabel("Exit date")
    ax.set_ylabel("Equity (starts at 1.0)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "equity_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(trades["exit_date"], trades["drawdown_pct"])
    ax.set_title("Final holdout drawdown (completed trades)")
    ax.set_xlabel("Exit date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "drawdown.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    strategy_pct = float(summary.get("strategy_compounded_net_return_pct", np.nan))
    benchmark_pct = float(summary.get("benchmark_net_return_pct", np.nan))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Strategy\n(compounded)", "Continuous-long\nbenchmark"], [strategy_pct, benchmark_pct])
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Final holdout net return comparison")
    ax.set_ylabel("Return (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "benchmark_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    acc = accuracy_score(y_true, y_pred) * 100
    bal = balanced_accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    win_rate = trades["net_return"].gt(0).mean() * 100 if len(trades) else np.nan
    max_dd = trades["drawdown_pct"].min() if len(trades) else np.nan
    excess = strategy_pct - benchmark_pct

    lines = [
        "# Candlestick AI — Final Holdout Diagnostic Report", "",
        f"- **Holdout:** {summary.get('holdout_start', 'N/A')} to {summary.get('holdout_end', 'N/A')}",
        f"- **Holdout rows:** {summary.get('holdout_rows', 'N/A')}",
        f"- **Eligible signal rows:** {summary.get('eligible_signal_rows', len(pred))}",
        f"- **Trades:** {len(trades)}", "",
        "## Classification", "",
        f"- Accuracy: **{acc:.2f}%**",
        f"- Balanced accuracy: **{bal:.2f}%**",
        f"- Macro F1: **{macro_f1:.4f}**",
        "- See `classification_report.csv`, `confusion_matrix.csv`, and `confusion_matrix.png` for per-class diagnostics.", "",
        "## Trading performance", "",
        f"- Strategy: **{summary.get('strategy', 'N/A')}**",
        f"- Confidence threshold: **{summary.get('confidence_threshold', 'N/A')}**",
        f"- Holding horizon: **{summary.get('horizon_sessions', 'N/A')} sessions**",
        f"- Assumed round-trip cost: **{summary.get('round_trip_cost_bps', 'N/A')} bps**",
        f"- Compounded net return (summary): **{pct(strategy_pct)}**",
        f"- Win rate: **{pct(win_rate)}**",
        f"- Profit factor recomputed from trades: **{pf(trades['net_return']):.3f}**" if np.isfinite(pf(trades["net_return"])) else "- Profit factor recomputed from trades: **N/A or infinite**",
        f"- Maximum drawdown based on completed trades: **{pct(max_dd)}**", "",
        "### By predicted pattern", "",
        "| Pattern | Trades | Win rate | Mean net | Median net | Compounded net | Profit factor |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_pattern.iterrows():
        lines.append(
            f"| {row['predicted_pattern']} | {int(row['trades'])} | {pct(row['win_rate_pct'])} | "
            f"{pct(row['mean_net_return_pct'])} | {pct(row['median_net_return_pct'])} | "
            f"{pct(row['compounded_net_return_pct'])} | {row['profit_factor']:.3f} |"
            if np.isfinite(row["profit_factor"]) else
            f"| {row['predicted_pattern']} | {int(row['trades'])} | {pct(row['win_rate_pct'])} | "
            f"{pct(row['mean_net_return_pct'])} | {pct(row['median_net_return_pct'])} | "
            f"{pct(row['compounded_net_return_pct'])} | N/A/infinite |"
        )
    lines += [
        "", "### By confidence band (descriptive only)", "",
        "| Confidence | Trades | Win rate | Mean net | Median net | Compounded net | Profit factor |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in by_confidence.iterrows():
        pf_text = f"{row['profit_factor']:.3f}" if np.isfinite(row["profit_factor"]) else "N/A/infinite"
        lines.append(
            f"| {row['confidence_band']} | {int(row['trades'])} | {pct(row['win_rate_pct'])} | "
            f"{pct(row['mean_net_return_pct'])} | {pct(row['median_net_return_pct'])} | "
            f"{pct(row['compounded_net_return_pct'])} | {pf_text} |"
        )
    lines += [
        "", "## Benchmark comparison", "",
        f"- Strategy compounded net return: **{pct(strategy_pct)}**",
        f"- Continuous-long benchmark net return: **{pct(benchmark_pct)}**",
        f"- Excess return versus benchmark: **{excess:.3f} percentage points**",
        f"- Benchmark: {summary.get('benchmark_note', 'See final_holdout_summary.json.')}", "",
        "![Equity curve](equity_curve.png)", "",
        "![Drawdown](drawdown.png)", "",
        "![Benchmark comparison](benchmark_comparison.png)", "",
        "## Assumptions and limitations", "",
        "- The strategy was frozen before the final holdout; do not tune against this period and continue to call it an untouched test.",
        "- Confidence bands are descriptive only and must not be used to choose a new threshold on this same holdout.",
        "- Pattern-class confidence is not the probability of a profitable trade.",
        "- The sample contains relatively few trades, so performance estimates are uncertain.",
        "- Equity and drawdown are calculated at completed-trade exits, not daily mark-to-market portfolio values.",
        "- The continuous-long benchmark excludes dividends and financing costs and uses one illustrative round-trip cost.",
        "- Results depend on the dataset, labels, execution assumptions, and modeled costs; this is not evidence of a live-trading edge.", "",
        "## Generated files", "",
        "- `classification_report.csv`, `confusion_matrix.csv`, `confusion_matrix.png`",
        "- `trades_by_pattern.csv`, `trades_by_confidence.csv`",
        "- `trades_with_equity_and_drawdown.csv`",
        "- `equity_curve.png`, `drawdown.png`, `benchmark_comparison.png`", "",
    ]
    (OUT / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Diagnostic report generated.")
    print(f"Markdown report: {OUT / 'diagnostic_report.md'}")
    print(f"All diagnostics: {OUT.resolve()}")
    print(f"Accuracy={acc:.2f}% | Balanced accuracy={bal:.2f}% | Macro F1={macro_f1:.4f}")
    print(f"Strategy={pct(strategy_pct)} | Benchmark={pct(benchmark_pct)} | Excess={excess:.3f} pp")


if __name__ == "__main__":
    main()
