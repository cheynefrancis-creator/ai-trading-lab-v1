# AI Trading Lab — Full Version 1

## What this is

A research-only trading dashboard. It downloads historical market data,
trains a machine-learning classifier, estimates the probability of a positive
future return, and backtests a simple threshold strategy against buy-and-hold.

**It does not place trades.**

## Easiest setup

### Option A — Run on a computer

1. Install Python 3.11 or newer.
2. Open a terminal in this folder.
3. Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

4. Streamlit will give you a local web address, usually:
   `http://localhost:8501`
5. Open that address in Chrome.

### Option B — Put it online

Upload this folder to a Python/Streamlit hosting service that supports
Streamlit applications. The app can then be opened from your phone like a
normal website.

## How to use it

1. Enter a ticker, e.g. `AAPL`.
2. Choose the history.
3. Choose the prediction horizon.
4. Choose the probability threshold.
5. Press **Run Full Analysis**.
6. Read the latest probability and the unseen-data backtest.
7. Download the CSV if you want to keep the results.

## Important

The backtest is deliberately simple. It is a research starting point, not
evidence that the strategy is profitable in live markets.

The next major improvement should be walk-forward validation, followed by
paper trading. Do not connect real money to this V1.
