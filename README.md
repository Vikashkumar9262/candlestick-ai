# Candlestick AI: CNN-Based Candlestick Pattern Classification

A deep learning project that uses a **1D Convolutional Neural Network (CNN)** to classify candlestick patterns in NIFTY 50 daily market data and investigates whether those predictions can support a trading strategy.

The project combines time-series feature engineering, deep learning, chronological model evaluation, and backtesting with transaction costs. Its main goal is not just to classify candlestick patterns, but also to examine whether those classifications provide useful information for trading.

> **Important finding:** The model achieved 63.92% classification accuracy on the final holdout dataset. However, the tested trading strategy returned approximately -0.41%, while the buy-and-hold benchmark returned approximately +12.63% after the modeled transaction cost. The classification results therefore do not establish a profitable trading edge.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Project Objectives](#3-project-objectives)
- [4. How It Works](#4-how-it-works)
- [5. Dataset](#5-dataset)
- [6. Candlestick Patterns](#6-candlestick-patterns)
- [7. Feature Engineering](#7-feature-engineering)
- [8. Model Architecture](#8-model-architecture)
- [9. Training and Evaluation Methodology](#9-training-and-evaluation-methodology)
- [10. Trading Strategy](#10-trading-strategy)
- [11. Final Holdout Results](#11-final-holdout-results)
- [12. Project Structure](#12-project-structure)
- [13. Installation](#13-installation)
- [14. How to Run](#14-how-to-run)
- [15. Technologies Used](#15-technologies-used)
- [16. Limitations](#16-limitations)
- [17. Future Improvements](#17-future-improvements)
- [18. Disclaimer](#18-disclaimer)

## 1. Project Overview

Candlestick AI explores the use of deep learning to recognize recurring candlestick formations in financial markets.

Candlestick charts represent price movements over a particular period using four main values:

- **Open:** The price at which the period begins.
- **High:** The highest price reached during the period.
- **Low:** The lowest price reached during the period.
- **Close:** The price at which the period ends.

Trading volume can provide additional information about market activity.

Traditional candlestick analysis uses predefined rules to identify formations such as Doji, Hammer, and Engulfing patterns. This project uses engineered numerical features and a CNN to learn relationships in recent candle sequences and predict one of six pattern classes.

The model's predictions are then evaluated separately from the classification task through a simple, predefined trading strategy.

### What makes this project useful?

It brings together several practical machine learning concepts:

- Financial time-series preprocessing
- Numerical feature engineering
- CNN-based classification
- Chronological validation
- Class-imbalance handling
- Confidence-threshold analysis
- Trading backtesting with transaction costs
- Final holdout testing and benchmark comparison

## 2. Problem Statement

Candlestick patterns are commonly used in technical analysis, but identifying a pattern does not automatically mean that a profitable trading opportunity exists.

A model may classify patterns correctly without predicting future returns accurately. Even when a signal is followed by a favorable price movement, transaction costs and adverse market conditions can eliminate its potential profitability.

This project investigates two separate questions:

1. Can a CNN learn to classify candlestick patterns from historical NIFTY 50 data?
2. Can the model's predictions generate useful trading results on unseen data after accounting for a modeled transaction cost?

Separating these questions helps avoid treating classification accuracy as proof of trading performance.

## 3. Project Objectives

The project has the following objectives:

1. Prepare historical NIFTY 50 OHLCV data for machine learning.
2. Convert raw price information into numerical candle features.
3. Use recent historical candles as input sequences.
4. Train a 1D CNN to classify six candlestick categories.
5. Evaluate the model using chronological data splits.
6. Test a predefined prediction-based trading strategy.
7. Measure returns, win rate, profit factor, and maximum drawdown.
8. Compare strategy performance against a simple buy-and-hold benchmark.
9. Document limitations and identify improvements for future research.

## 4. How It Works

The project follows this workflow:

1. **Load historical data:** Read the NIFTY 50 OHLCV dataset.
2. **Preprocess the data:** Calculate candle characteristics and other numerical features.
3. **Create sequences:** Use the previous 30 candles as the model input.
4. **Train the CNN:** Learn relationships between the sequence features and pattern labels.
5. **Evaluate predictions:** Measure classification performance on later, unseen periods.
6. **Generate trading signals:** Apply the predefined pattern and confidence rules.
7. **Backtest the strategy:** Calculate returns using the specified entry, exit, and transaction-cost assumptions.
8. **Compare results:** Evaluate the strategy against the benchmark and interpret its limitations.

## 5. Dataset

The project uses historical daily NIFTY 50 market data.

| Property | Value |
|---|---|
| Instrument | NIFTY 50 index |
| Data frequency | Daily |
| Dataset start | 2010-01-04 |
| Dataset end | 2025-12-31 |
| Total rows | 3,929 |
| Core data | Open, High, Low, Close, Volume |
| Input sequence length | 30 candles |
| Prediction task | Six-class classification |

The dataset is stored at:

`data/raw/nifty50.csv`

The model uses historical sequences to classify candlestick patterns. The final evaluation uses the latest portion of the dataset, which is kept separate from model training and validation.

**Data note:** The exact provenance, licensing, and adjustment methodology of the CSV should be documented alongside the dataset before publishing the project.

## 6. Candlestick Patterns

The model predicts one of six classes.

| Pattern | General interpretation |
|---|---|
| `none` | No recognized target pattern |
| `doji` | The opening and closing prices are relatively close |
| `hammer` | A candle with a relatively long lower wick |
| `shooting_star` | A candle with a relatively long upper wick |
| `bullish_engulfing` | A bullish candle that engulfs the preceding candle's body according to the labeling rules |
| `bearish_engulfing` | A bearish candle that engulfs the preceding candle's body according to the labeling rules |

These names describe pattern categories, not guaranteed market outcomes.

The exact rules used to assign labels—including wick, body, and engulfing thresholds—are determined by the dataset-labeling implementation. They should be checked in the source code when reproducing or modifying the experiment.

## 7. Feature Engineering

Raw OHLCV values are transformed into seven numerical features.

| Feature | Explanation |
|---|---|
| `body_ratio` | Candle-body size relative to its overall range |
| `upper_wick_ratio` | Upper-wick size relative to the candle range |
| `lower_wick_ratio` | Lower-wick size relative to the candle range |
| `candle_direction` | Numerical representation of bullish or bearish direction |
| `close_return` | Relative change in closing price |
| `range_return` | Relative measure of the candle's price range |
| `relative_volume` | Volume compared with a reference level |

These features help the model represent candle shape, direction, recent price movement, and trading activity.

### Why use engineered features?

The raw market data contains price levels that can vary considerably over time. Ratios and returns can express candle characteristics in a more comparable numerical form.

The feature scaler is fitted using training data rather than the final holdout data. This reduces the risk of information leakage during preprocessing.

## 8. Model Architecture

The project uses a **1D Convolutional Neural Network (CNN)** implemented in PyTorch.

A 1D CNN applies filters across ordered sequence data. Here, the sequence represents the engineered features of recent daily candles.

### Model workflow

1. Input the feature sequence containing the previous 30 candles.
2. Apply a convolutional layer with 32 channels and a kernel size of 3.
3. Apply ReLU activation.
4. Apply a second convolutional layer with 64 channels and a kernel size of 3.
5. Apply another ReLU activation.
6. Combine the latest convolutional representation with selected raw features from the latest candle.
7. Pass the combined representation through a 64-unit fully connected layer.
8. Apply ReLU and dropout.
9. Produce six output scores, one for each candlestick class.

The output scores are converted into class probabilities for prediction and confidence-threshold analysis.

### Training configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Batch size | 64 |
| Maximum epochs | 40 |
| Dropout | 0.30 |
| Weight decay | 0.0001 |
| Early stopping patience | 6 epochs |
| Random seed | 42 |
| Loss weighting | Balanced class weights in walk-forward training |

Early stopping helps limit unnecessary training when validation loss stops improving.

## 9. Training and Evaluation Methodology

Financial data is time-dependent. Randomly splitting individual observations into training and testing sets can allow future information to influence the evaluation.

For this reason, the project uses chronological splits.

### Final holdout split

| Dataset portion | Approximate row range | Dates |
|---|---|---|
| Training | 0–3181 | Through 2022-12-21 |
| Validation | 3187–3535 | 2022-12-29 to 2024-06-03 |
| Final holdout | 3536–3928 | 2024-06-04 to 2025-12-31 |

A five-row purge gap is used around the training and validation boundaries to help reduce leakage associated with overlapping observations and sequence construction.

The final holdout contains 393 daily rows. Because a complete five-session forward return is required for each strategy trade, not every row can be used as a trade signal.

### Evaluation metrics

- **Accuracy:** The proportion of predictions that match the true class.
- **Balanced accuracy:** The average recall across classes, helping account for class imbalance.
- **Macro F1:** The average F1 score across classes, giving each class equal weight.
- **Win rate:** The proportion of trades with positive net returns.
- **Profit factor:** Gross profits divided by the absolute value of gross losses.
- **Maximum drawdown:** The largest peak-to-trough decline in the strategy's equity curve.
- **Compounded return:** The return obtained by compounding the sequence of trade returns.

Classification metrics measure pattern recognition. Trading metrics measure the outcomes of the specific strategy tested. They answer different questions.

## 10. Trading Strategy

After classification, the project evaluates a frozen, bullish-only strategy.

### Strategy rules

| Rule | Specification |
|---|---|
| Strategy | Bullish-only |
| Eligible predicted patterns | `bullish_engulfing`, `hammer` |
| Confidence threshold | At least 70% |
| Holding horizon | Five trading sessions |
| Signal timing | After the signal candle closes |
| Entry | Next session's Open |
| Exit | Close on the fifth session after the signal |
| Trade overlap | No overlapping trades |
| Modeled transaction cost | 10 basis points round-trip |

A basis point is one-hundredth of one percentage point. Therefore, 10 basis points equals 0.10%.

### Example

Suppose the model identifies a bullish pattern with a predicted probability of 82%.

- The signal is eligible because the probability exceeds the 70% threshold.
- The strategy enters at the next session's opening price.
- It exits at the specified fifth-session closing price.
- The modeled transaction cost is deducted from the trade return.

The strategy does not enter a trade merely because a pattern is detected; the confidence threshold and no-overlap rule must also be satisfied.

### Benchmark

The strategy is compared with a continuous long benchmark over the same final holdout period.

The benchmark holds the index from the first eligible holdout Open to the final Close, with one illustrative round-trip transaction cost. The comparison excludes dividends and financing costs.

This is a simple reference benchmark, not a fully investable portfolio simulation.

## 11. Final Holdout Results

The final holdout was evaluated after the strategy rules had been frozen.

### Classification performance

| Metric | Result |
|---|---:|
| Accuracy | 63.92% |
| Balanced accuracy | 88.91% |
| Macro F1 | 51.70% |

Balanced accuracy is higher than ordinary accuracy because it gives each class equal weight. This difference can indicate uneven class-level performance, so the confusion matrix and per-class precision and recall should also be reviewed.

### Trading performance

| Metric | Strategy | Buy-and-hold benchmark |
|---|---:|---:|
| Net return | -0.41% | +12.63% |
| Number of strategy trades | 36 | Not applicable |
| Strategy win rate | 52.78% | Not applicable |
| Strategy profit factor | 0.996 | Not applicable |
| Strategy maximum drawdown | -8.11% | Not reported here |

The strategy's compounded net return was approximately -0.41%. Its profit factor was slightly below 1, indicating that gross losses marginally exceeded gross profits.

The benchmark returned approximately +12.63% after the modeled transaction cost. The strategy therefore underperformed the benchmark by approximately **13.04 percentage points**.

### What do these results mean?

The results show that the model can recognize some candlestick patterns, but the tested strategy did not demonstrate a profitable trading edge on the final holdout.

A classification accuracy of 63.92% does not mean that 63.92% of trades are profitable. Pattern labels and future price returns are different targets.

The result also does not prove that every possible strategy using candlestick patterns is unprofitable. It shows that this particular model, confidence threshold, holding period, and execution setup did not outperform the chosen benchmark in the tested period.

## 12. Project Structure

The main project files are organized as follows:

```text
candlestick-ai/
├── data/
│   ├── raw/
│   │   └── nifty50.csv
│   └── processed/
├── models/
│   └── final_holdout/
│       ├── candlestick_cnn.pt
│       ├── scaler.joblib
│       └── metadata.json
├── reports/
│   ├── walk_forward_predictions.csv
│   ├── strategy_research_summary.csv
│   ├── strategy_research_trades.csv
│   ├── final_holdout_predictions.csv
│   ├── final_holdout_trades.csv
│   ├── final_holdout_summary.json
│   └── frozen_strategy_spec.md
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── walk_forward.py
│   ├── strategy_research.py
│   ├── analyze_strategy_folds.py
│   ├── analyze_cost_sensitivity.py
│   ├── validate_strategy_returns.py
│   └── final_holdout.py
├── tests/
├── requirements.txt
└── README.md
```

The structure above reflects the files used in the project workflow. Some optional diagnostic or analysis scripts may be added separately as development continues.

### Important files

- `src/dataset.py`: Prepares sequences, features, labels, and scaling.
- `src/model.py`: Defines the CNN architecture.
- `src/train.py`: Contains model-training logic.
- `src/evaluate.py`: Supports model evaluation.
- `src/walk_forward.py`: Runs chronological walk-forward evaluation.
- `src/strategy_research.py`: Tests candidate trading rules.
- `src/analyze_strategy_folds.py`: Examines strategy performance across folds.
- `src/analyze_cost_sensitivity.py`: Tests the effect of transaction costs.
- `src/validate_strategy_returns.py`: Validates calculated trade returns against raw price data.
- `src/final_holdout.py`: Runs the final holdout experiment.
- `reports/`: Stores predictions, trade records, and evaluation summaries.
- `models/final_holdout/`: Stores the final trained model and associated preprocessing metadata.

## 13. Installation

### Prerequisites

- Python compatible with the project's dependencies
- Git, if cloning the repository
- A terminal such as Windows PowerShell
- The historical dataset file

### Step 1: Clone the repository

Replace the placeholder URL with the actual repository URL.

```bash
git clone <YOUR_REPOSITORY_URL>
cd candlestick-ai
```

Alternatively, open a terminal inside your existing project folder.

### Step 2: Create a virtual environment

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure `requirements.txt` contains the dependencies required by the actual project. PyTorch, pandas, NumPy, scikit-learn, and joblib are among the libraries used by the workflow.

### Step 4: Check the dataset

Confirm that this file exists:

```text
data/raw/nifty50.csv
```

The CSV must contain the columns expected by the dataset-loading code. Check `src/dataset.py` for the exact column names and preprocessing assumptions.

## 14. How to Run

Run commands from the repository root with the virtual environment activated.

### Run walk-forward evaluation

```bash
python -m src.walk_forward
```

This evaluates the model across chronological folds and writes walk-forward predictions and evaluation results.

### Research candidate trading strategies

```bash
python -m src.strategy_research
```

This evaluates the configured candidate strategies, confidence thresholds, and transaction-cost assumptions.

### Analyze strategy performance across folds

```bash
python -m src.analyze_strategy_folds
```

### Analyze transaction-cost sensitivity

```bash
python -m src.analyze_cost_sensitivity
```

### Validate strategy return calculations

```bash
python -m src.validate_strategy_returns
```

### Run the final holdout experiment

```bash
python -m src.final_holdout
```

The final holdout script trains and evaluates the model using the configured chronological split, then saves the resulting predictions, trade records, model artifacts, and summary metrics.

**Reproducibility note:** Command names, required arguments, and output paths should be verified against the current scripts before publishing. Do not run the final holdout repeatedly while tuning the model and then treat the same holdout as an untouched test set.

## 15. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | CNN implementation and training |
| pandas | Tabular data processing |
| NumPy | Numerical operations |
| scikit-learn | Preprocessing and evaluation utilities |
| joblib | Saving preprocessing objects |
| pytest | Automated testing |
| CSV and JSON | Storing results and experiment summaries |
| Git | Version control, if configured |

## 16. Limitations

The project has several important limitations.

### 1. Limited evidence of a trading edge

The frozen strategy lost approximately 0.41% on the final holdout and underperformed the benchmark. It should not be presented as a profitable trading system.

### 2. Small final strategy sample

The final holdout generated only 36 trades. A small sample creates uncertainty around win rate, profit factor, and expected returns.

### 3. Transaction costs are simplified

The backtest uses a fixed round-trip cost of 10 basis points. Actual trading costs may include brokerage, taxes, exchange charges, slippage, bid-ask spreads, and market impact.

### 4. Execution assumptions are idealized

The strategy uses the next session's Open as the entry and a specified future Close as the exit. Real-world execution may differ because of liquidity, gaps, order timing, and price availability.

### 5. Benchmark limitations

The benchmark excludes dividends and financing costs. It is useful as a reference but does not represent every feature of a real investment portfolio.

### 6. Classification and profitability are different objectives

The model learns to predict predefined pattern labels. It is not directly trained to maximize risk-adjusted trading returns.

### 7. Historical patterns may not remain stable

Financial market behavior can change over time. Relationships learned from historical data may weaken or disappear in future market conditions.

### 8. Dataset and labeling assumptions matter

Results depend on the quality of the source data, the exact labeling rules, missing-value handling, and feature calculations. These should be documented and independently verified.

## 17. Future Improvements

Potential next steps include:

1. **Per-class evaluation:** Report precision, recall, F1, and confusion matrices for all six classes.
2. **Calibration analysis:** Test whether predicted probabilities correspond to observed classification frequencies.
3. **Stronger baselines:** Compare the CNN with simple rule-based pattern recognition, logistic regression, tree-based models, and naive predictors.
4. **Robust walk-forward testing:** Evaluate across more time periods and market regimes.
5. **Cost and slippage modeling:** Include realistic fees, spreads, and adverse execution.
6. **Risk-adjusted metrics:** Report volatility, Sharpe ratio, exposure, turnover, and benchmark-relative drawdown where appropriate.
7. **Statistical uncertainty:** Use confidence intervals and robustness tests for strategy returns.
8. **Ablation studies:** Determine whether volume, wick ratios, or other features contribute meaningful predictive information.
9. **Alternative targets:** Investigate future returns or risk-adjusted outcomes directly rather than relying exclusively on pattern labels.
10. **Paper trading:** If later results justify further investigation, test the system without real capital before considering live deployment.

Future changes should be evaluated on new, unseen data rather than repeatedly optimizing against the existing final holdout.

## 18. Disclaimer

This repository is an educational machine learning and quantitative research project. It is not financial advice, and it is not a recommendation to buy or sell any financial instrument.

Historical performance does not guarantee future results. The reported backtest uses simplified cost and execution assumptions, and the tested strategy did not outperform its benchmark during the final holdout period.

Do not use the model for live trading without substantially more validation, realistic execution modeling, risk controls, and independent testing.

---

## Conclusion

Candlestick AI demonstrates how a deep learning model can be built and evaluated for candlestick-pattern classification using historical financial time-series data.

The project also demonstrates why a machine learning model must be evaluated beyond its classification metrics. Although the CNN recognized patterns with measurable classification performance, the tested trading strategy failed to produce a positive net return on the final holdout and substantially underperformed the benchmark.

That result is an important part of the research: **a model that recognizes a pattern is not necessarily a model that predicts a profitable trade.**
