import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["open", "high", "low", "close", "volume", "log_ret_1"]
TARGET_COLS = ["target_1", "target_5", "target_21"]


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


def rows_flat(values_t: np.ndarray, indices: np.ndarray, window: int) -> np.ndarray:
    idx = indices.astype(np.int64)
    out = np.stack([values_t[i - window + 1 : i + 1].reshape(-1) for i in idx])
    return out.astype(np.float64)


def save_artifacts(
    out_dir: Path,
    model: MultiOutputRegressor,
    scaler: StandardScaler,
    window: int,
    feat_names: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    meta = {
        "window": window,
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "flat_feature_names": feat_names,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/AAPL.csv")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument(
        "--artifacts",
        default=None,
        help="Directory to write model.joblib, scaler.joblib, meta.json after training",
    )
    args = p.parse_args()

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
        raise SystemExit("Splits too small")

    train_i = all_i[:n_train]
    test_i = all_i[n_train + n_val :]

    scaler = fit_scaler(values, train_i, w)
    values_t = transform_values(values, scaler)

    X_train = rows_flat(values_t, train_i, w)
    X_test = rows_flat(values_t, test_i, w)
    feat_names = [str(j) for j in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feat_names)
    X_test_df = pd.DataFrame(X_test, columns=feat_names)

    y_train = targets[train_i]
    y_test = targets[test_i]

    base = LGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=args.seed,
        verbosity=-1,
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train_df, y_train)

    y_hat = model.predict(X_test_df)
    rmse = root_mean_squared_error(y_test, y_hat, multioutput="raw_values")
    baseline = np.zeros_like(y_test)
    rmse_base = root_mean_squared_error(y_test, baseline, multioutput="raw_values")

    names = ",".join(TARGET_COLS)
    print(f"lgbm_test_rmse ({names}): {rmse}")
    print(f"baseline_rmse ({names}): {rmse_base}")

    if args.artifacts:
        save_artifacts(Path(args.artifacts), model, scaler, w, feat_names)
        print(f"artifacts_saved: {args.artifacts}")


if __name__ == "__main__":
    main()
