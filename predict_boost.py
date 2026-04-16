import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", required=True)
    p.add_argument(
        "--data",
        required=True,
        help="CSV with date index and the same feature columns as training (e.g. prepare_data output)",
    )
    args = p.parse_args()

    root = Path(args.artifacts)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    model = joblib.load(root / "model.joblib")
    scaler = joblib.load(root / "scaler.joblib")

    w = int(meta["window"])
    feat_cols = meta["feature_cols"]
    target_cols = meta["target_cols"]
    flat_names = meta["flat_feature_names"]

    df = pd.read_csv(Path(args.data), index_col="date", parse_dates=True)
    df = df.sort_index()
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    if df[feat_cols].isna().any().any():
        raise SystemExit("NaNs in features")

    n = len(df)
    if n < w:
        raise SystemExit(f"Need at least {w} rows, got {n}")

    values = df[feat_cols].to_numpy(dtype=np.float64)
    vt = scaler.transform(values.reshape(-1, values.shape[1]))
    values_t = vt.reshape(values.shape).astype(np.float32)
    end = n - 1
    x = values_t[end - w + 1 : end + 1].reshape(1, -1)
    X = pd.DataFrame(x, columns=flat_names)
    pred = model.predict(X)[0]

    asof = df.index[-1]
    print(f"as_of: {asof}")
    for name, val in zip(target_cols, pred):
        print(f"  {name}: {val:.6f}")


if __name__ == "__main__":
    main()
