import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def predict_one(
    df: pd.DataFrame,
    model,
    scaler,
    window: int,
    feat_cols: list[str],
    flat_names: list[str],
) -> np.ndarray:
    n = len(df)
    end = n - 1
    values = df[feat_cols].to_numpy(dtype=np.float64)
    vt = scaler.transform(values.reshape(-1, values.shape[1]))
    values_t = vt.reshape(values.shape).astype(np.float32)
    x = values_t[end - window + 1 : end + 1].reshape(1, -1)
    X = pd.DataFrame(x, columns=flat_names)
    return model.predict(X)[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", required=True)
    p.add_argument("--data", required=True)
    p.add_argument(
        "--as-of",
        default=None,
        help="Trading day used as the last bar in the window (default: last date in CSV)",
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
    missing = [c for c in feat_cols + target_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    if args.as_of is None:
        sub = df
        asof = df.index[-1]
    else:
        asof = pd.Timestamp(args.as_of)
        if asof not in df.index:
            raise SystemExit(f"as_of not in index: {asof}")
        sub = df.loc[:asof]
    if sub[feat_cols].isna().any().any():
        raise SystemExit("NaNs in features for chosen slice")
    if len(sub) < w:
        raise SystemExit(f"Need at least {w} rows through as_of, got {len(sub)}")

    pred = predict_one(sub, model, scaler, w, feat_cols, flat_names)
    actual = df.loc[asof, target_cols].to_numpy(dtype=np.float64)
    if np.isnan(actual).any():
        raise SystemExit("Actual targets not available yet for this as_of (NaN in CSV)")

    print(f"as_of: {asof}")
    for name, pval, aval in zip(target_cols, pred, actual):
        err = pval - aval
        dir_ok = bool(np.sign(pval) == np.sign(aval))
        print(
            f"{name}: pred={pval:.6f} actual={aval:.6f} err={err:.6f} dir_ok={dir_ok}"
        )


if __name__ == "__main__":
    main()
