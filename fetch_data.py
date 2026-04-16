import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="AAPL")
    p.add_argument("--period", default="10y")
    p.add_argument(
        "--out",
        default=None,
        help="CSV path (default: data/raw/{symbol}.csv)",
    )
    args = p.parse_args()

    out = Path(args.out) if args.out else Path("data/raw") / f"{args.symbol.upper()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    df = yf.Ticker(args.symbol).history(period=args.period, auto_adjust=True, actions=False)
    if df.empty:
        raise SystemExit(f"No data for {args.symbol!r}")

    df = df.sort_index()
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].dropna(how="any")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df.to_csv(out)


if __name__ == "__main__":
    main()
