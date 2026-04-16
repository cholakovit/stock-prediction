import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/raw/AAPL.csv")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.out) if args.out else Path("data/processed") / inp.name

    df = pd.read_csv(inp, index_col="date", parse_dates=True)
    df = df.sort_index()
    close = df["close"].astype(np.float64)

    for h in (1, 5, 21):
        df[f"target_{h}"] = np.log(close.shift(-h)) - np.log(close)

    df["log_ret_1"] = np.log(close / close.shift(1))

    need = ["target_1", "target_5", "target_21", "log_ret_1"]
    df = df.dropna(subset=need)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)

if __name__ == "__main__":
    main()
