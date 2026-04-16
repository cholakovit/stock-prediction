import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


FEATURE_COLS = ["open", "high", "low", "close", "volume", "log_ret_1"]
TARGET_COLS = ["target_1", "target_5", "target_21"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StockWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        targets: np.ndarray,
        window: int,
        indices: np.ndarray,
    ) -> None:
        self.values = values
        self.targets = targets
        self.window = window
        self.indices = indices

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(self.indices[idx])
        s = i - self.window + 1
        x = self.values[s : i + 1].astype(np.float32)
        y = self.targets[i].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        d = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers,
            batch_first=True,
            dropout=d,
        )
        self.head = nn.Linear(hidden, len(TARGET_COLS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def fit_scaler(values: np.ndarray, indices: np.ndarray, window: int) -> StandardScaler:
    parts = []
    for i in indices:
        s = int(i) - window + 1
        parts.append(values[s : int(i) + 1])
    flat = np.vstack(parts).astype(np.float64)
    scaler = StandardScaler()
    scaler.fit(flat)
    return scaler


def transform_values(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    t = scaler.transform(values.reshape(-1, values.shape[-1]))
    return t.reshape(values.shape).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/AAPL.csv")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    df = pd.read_csv(Path(args.data), index_col="date", parse_dates=True)
    df = df.sort_index()
    if df[FEATURE_COLS + TARGET_COLS].isna().any().any():
        raise SystemExit("NaNs in features or targets")

    values = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    targets = df[TARGET_COLS].to_numpy(dtype=np.float64)
    n = len(df)
    w = args.window
    if n < w:
        raise SystemExit("Not enough rows for window")

    all_i = np.arange(w - 1, n, dtype=np.int64)
    m = all_i.shape[0]
    n_train = int(0.7 * m)
    n_val = int(0.15 * m)
    n_test = m - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise SystemExit("Splits too small; need more data or smaller window")

    train_i = all_i[:n_train]
    val_i = all_i[n_train : n_train + n_val]
    test_i = all_i[n_train + n_val :]

    scaler = fit_scaler(values, train_i, w)
    values_t = transform_values(values, scaler)

    train_ds = StockWindowDataset(values_t, targets, w, train_i)
    val_ds = StockWindowDataset(values_t, targets, w, val_i)
    test_ds = StockWindowDataset(values_t, targets, w, test_i)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(
        n_features=len(FEATURE_COLS),
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best = float("inf")
    bad = 0
    best_state = None

    for _ in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
        val_mse = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_mse < best - 1e-9:
            best = val_mse
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(dev)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            actuals.append(yb.numpy())
    y_hat = np.vstack(preds)
    y_true = np.vstack(actuals)

    rmse = root_mean_squared_error(y_true, y_hat, multioutput="raw_values")
    baseline = np.zeros_like(y_true)
    rmse_base = root_mean_squared_error(y_true, baseline, multioutput="raw_values")

    names = ",".join(TARGET_COLS)
    print(f"test_rmse ({names}): {rmse}")
    print(f"baseline_rmse ({names}): {rmse_base}")


if __name__ == "__main__":
    main()
