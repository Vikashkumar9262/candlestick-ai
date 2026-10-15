
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import (
    FEATURES, CLASSES, LABEL_TO_ID,
    make_features, make_labels, make_sequences,
)
from src.model import CandlestickCNN

DATA_PATH = Path("data/raw/nifty50.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 30
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():
    df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    df = df.sort_index()

    # Features use current and past candles only.
    raw_features = make_features(df)
    if "Volume" not in df.columns:
        raw_features["relative_volume"] = 1.0
    else:
        # A neutral fallback for unavailable/zero-volume history.
        raw_features["relative_volume"] = (
            raw_features["relative_volume"].fillna(1.0)
        )

    raw_features = raw_features.replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    labels = make_labels(df).map(LABEL_TO_ID).to_numpy()

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Fit the scaler on training-period rows only.
    scaler = StandardScaler()
    scaler.fit(raw_features.iloc[:train_end][FEATURES])

    scaled = scaler.transform(raw_features[FEATURES])
    X, y, idx = make_sequences(scaled, labels, WINDOW)

    train_mask = idx < train_end
    val_mask = (idx >= train_end) & (idx < val_end)
    test_mask = idx >= val_end

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("Device:", device)
    print("Samples:", len(X_train), len(X_val), len(X_test))
    print("Training classes:", np.bincount(y_train, minlength=len(CLASSES)))

    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    present_classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train,
    )
    class_weights = np.ones(len(CLASSES), dtype=np.float32)
    class_weights[present_classes] = weights

    model = CandlestickCNN(
        num_features=len(FEATURES),
        num_classes=len(CLASSES),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )

    best_val_loss = float("inf")
    patience = 6
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(xb)

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                val_loss_sum += loss.item() * len(xb)

        train_loss = train_loss_sum / max(len(train_ds), 1)
        val_loss = val_loss_sum / max(len(val_ds), 1)

        print(
            f"Epoch {epoch + 1:02d}: "
            f"train={train_loss:.4f}, val={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                model.state_dict(),
                MODEL_DIR / "candlestick_cnn.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping.")
            break

    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(
        {
            "features": FEATURES,
            "classes": CLASSES,
            "window": WINDOW,
            "train_end": str(df.index[train_end - 1]),
            "val_end": str(df.index[val_end - 1]),
        },
        MODEL_DIR / "metadata.joblib",
    )

    print("Best validation loss:", best_val_loss)
    print("Saved model, scaler, and metadata.")


if __name__ == "__main__":
    main()
