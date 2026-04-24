TASK: A market research and investment support team wants to anticipate short-, medium-, and monthly movement patterns for a major publicly traded company so they can improve planning, compare likely outcomes across time horizons, and support faster, more consistent decisions around timing, risk awareness, and portfolio discussion priorities.

------------------------------------------------------------

## Goal

Learn a sequence model (LSTM) on historical daily prices for one liquid symbol (e.g. AAPL), then forecast cumulative log-returns at three horizons: 1 trading day, 5 trading days (~week), 21 trading days (~month). Horizons are defined by **trading-day offsets**, not calendar months.

**Repository:** [github.com/cholakovit/stock-prediction](https://github.com/cholakovit/stock-prediction)

## Inputs

For each date `t`, the model uses a fixed window of the past `L` trading days (e.g. adjusted close, volume; optionally returns inside the window). No future data in the input.

## Targets

Let `P_t` be adjusted close on trading day `t`. Log-return from `t` to `t+h`:

`r_t^(h) = log(P_{t+h}) - log(P_t)`

- **Day:** `h = 1`
- **Week:** `h = 5` (five trading days)
- **Month:** `h = 21` (~one month of trading days; pick 20 or 22 if you prefer and keep README and code aligned)

**Alternative:** binary direction per horizon: `1` if `r_t^(h) > 0` else `0` (three heads or three separate models).

## Splits

Train / validation / test by **time** (e.g. 70% / 15% / 15% chronological). Fit scalers **only** on train.

## Metrics

Regression: MAE / RMSE per horizon. Classification: accuracy / F1 per horizon. Include a **naive baseline** (e.g. predict 0 return) on the same test window.

## Scope

Single symbol, daily bars, no costs, no live-trading claim. Demonstration of methodology only, not investment advice.

## Setup

Python 3.10+ recommended. Create a venv, then:

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
python fetch_data.py --symbol AAPL --period 10y
python prepare_data.py
python train.py
python train_boost.py
python predict_boost.py
python verify_boost.py
```

`data/` and `artifacts/` are gitignored; run `fetch_data` and `prepare_data` after clone. LightGBM models land under `artifacts/lgbm/` after `train_boost.py`.
