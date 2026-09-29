import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError:
    yf = None

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(
    page_title="AI Trading Lab V1",
    page_icon="📈",
    layout="wide",
)

FEATURES = [
    "ret_1", "ret_5", "ret_20", "vol_20", "rsi_14",
    "sma_10_gap", "sma_50_gap", "sma_200_gap",
    "range_pct", "volume_z"
]

@st.cache_data(ttl=3600)
def download_data(ticker: str, period: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed.")
    df = yf.download(
        ticker, period=period, interval="1d",
        auto_adjust=True, progress=False
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[required].dropna()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def engineer(df, horizon):
    x = df.copy()
    c = x["Close"]
    ret = c.pct_change()

    x["ret_1"] = ret
    x["ret_5"] = c.pct_change(5)
    x["ret_20"] = c.pct_change(20)
    x["vol_20"] = ret.rolling(20).std()
    x["rsi_14"] = rsi(c)
    x["sma_10_gap"] = c / c.rolling(10).mean() - 1
    x["sma_50_gap"] = c / c.rolling(50).mean() - 1
    x["sma_200_gap"] = c / c.rolling(200).mean() - 1
    x["range_pct"] = (x["High"] - x["Low"]) / c
    x["volume_z"] = (
        (x["Volume"] - x["Volume"].rolling(20).mean())
        / x["Volume"].rolling(20).std()
    )

    # Future return is only used to create the target/backtest outcome.
    x["future_return"] = c.shift(-horizon) / c - 1
    x["target"] = (x["future_return"] > 0).astype(int)

    return x.dropna(subset=FEATURES + ["future_return"]).copy()

def train_and_test(data, train_fraction):
    split = int(len(data) * train_fraction)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train[FEATURES], train["target"])
    test["prob_up"] = model.predict_proba(test[FEATURES])[:, 1]
    test["prediction"] = (test["prob_up"] >= 0.50).astype(int)

    auc = roc_auc_score(test["target"], test["prob_up"])
    accuracy = accuracy_score(test["target"], test["prediction"])
    return model, train, test, auc, accuracy

def backtest(test, threshold, cost_bps):
    b = test.copy()
    b["signal"] = (b["prob_up"] >= threshold).astype(int)
    cost = cost_bps / 10000.0

    # Conservative simplification: enter only on qualifying signals.
    b["strategy_return"] = np.where(
        b["signal"] == 1,
        b["future_return"] - cost,
        0.0
    )
    b["buy_hold_return"] = b["future_return"]

    b["strategy_equity"] = (1 + b["strategy_return"]).cumprod()
    b["buy_hold_equity"] = (1 + b["buy_hold_return"]).cumprod()

    peak = b["strategy_equity"].cummax()
    b["drawdown"] = b["strategy_equity"] / peak - 1

    return b

def feature_reading(latest):
    readings = []
    if latest["rsi_14"] >= 70:
        readings.append(("RSI", "Overbought zone"))
    elif latest["rsi_14"] <= 30:
        readings.append(("RSI", "Oversold zone"))
    else:
        readings.append(("RSI", "Neutral zone"))

    readings.append((
        "Trend",
        "Above 50/200-day averages"
        if latest["sma_50_gap"] > 0 and latest["sma_200_gap"] > 0
        else "Mixed / below one or more major averages"
    ))
    readings.append((
        "Momentum",
        "Positive"
        if latest["ret_5"] > 0 and latest["ret_20"] > 0
        else "Mixed / negative"
    ))
    readings.append((
        "Volume",
        "Above recent average"
        if latest["volume_z"] > 0 else "At/below recent average"
    ))
    return readings

st.title("📈 AI Trading Lab — Full Version 1")
st.caption("Research dashboard • Machine learning • Backtesting • No live trading")

with st.sidebar:
    st.header("Research controls")
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    period = st.selectbox("Historical data", ["5y", "10y", "max"], index=1)
    horizon = st.selectbox(
        "Prediction horizon (trading days)",
        [1, 3, 5, 10, 20],
        index=2
    )
    threshold = st.slider(
        "Research signal threshold",
        0.50, 0.80, 0.60, 0.01
    )
    train_fraction = st.slider(
        "Training share",
        0.60, 0.85, 0.70, 0.05
    )
    cost_bps = st.number_input(
        "Assumed round-trip cost (basis points)",
        min_value=0, max_value=200, value=10, step=1
    )
    run = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

if not run:
    st.info("Choose your settings and press **Run Full Analysis**.")
    st.markdown("""
### What this V1 does

1. Downloads historical daily market data.
2. Creates momentum, volatility, RSI, moving-average and volume features.
3. Trains a machine-learning classifier on the earlier part of history.
4. Tests it on later data it did **not** train on.
5. Estimates the latest probability of a positive move.
6. Simulates a simple threshold strategy.
7. Compares that strategy with buy-and-hold.

### Safety by design

**This version cannot place a trade.** There is no broker connection, API key field, deposit function or automatic order execution.

The purpose is to find out whether the statistical idea has evidence behind it before real money is considered.
""")
    st.stop()

try:
    with st.spinner(f"Analysing {ticker}..."):
        raw = download_data(ticker, period)
        data = engineer(raw, horizon)

        if len(data) < 300:
            raise ValueError(
                "Not enough usable history. Try a longer history period."
            )

        model, train, test, auc, accuracy = train_and_test(
            data, train_fraction
        )
        bt = backtest(test, threshold, cost_bps)

        latest_row = data.iloc[[-1]]
        latest = data.iloc[-1]
        latest_prob = float(model.predict_proba(latest_row[FEATURES])[0, 1])

    st.success(f"Analysis complete: {ticker}")

    # Headline metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model probability", f"{latest_prob:.1%}")
    c2.metric("Out-of-sample AUC", f"{auc:.3f}")
    c3.metric("Out-of-sample accuracy", f"{accuracy:.1%}")
    c4.metric("Unseen test rows", f"{len(test):,}")

    if latest_prob >= threshold:
        st.success(
            f"🟢 Research signal: probability {latest_prob:.1%} "
            f"is above your {threshold:.0%} threshold."
        )
    else:
        st.info(
            f"⚪ No threshold signal: probability {latest_prob:.1%} "
            f"is below your {threshold:.0%} threshold."
        )

    st.warning(
        "Research output only. This probability is an estimate, not a guarantee "
        "or a recommendation to buy or sell."
    )

    st.subheader("Why did the model see this setup?")
    for name, reading in feature_reading(latest):
        st.write(f"**{name}:** {reading}")

    # Backtest metrics
    active = bt[bt["signal"] == 1]
    strategy_return = float(bt["strategy_equity"].iloc[-1] - 1)
    buy_hold_return = float(bt["buy_hold_equity"].iloc[-1] - 1)
    max_dd = float(bt["drawdown"].min())
    win_rate = (
        float((active["future_return"] > 0).mean())
        if len(active) else np.nan
    )

    st.subheader("🧪 Unseen-data backtest")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AI strategy", f"{strategy_return:.2%}")
    m2.metric("Buy & hold", f"{buy_hold_return:.2%}")
    m3.metric("Signals", f"{len(active):,}")
    m4.metric(
        "Signal win rate",
        "—" if np.isnan(win_rate) else f"{win_rate:.1%}"
    )
    m5.metric("Max drawdown", f"{max_dd:.2%}")

    st.write(
        f"Training period: **{train.index.min().date()} → "
        f"{train.index.max().date()}**  \n"
        f"Unseen test period: **{test.index.min().date()} → "
        f"{test.index.max().date()}**"
    )

    fig = plt.figure(figsize=(12, 5))
    plt.plot(
        bt.index, bt["strategy_equity"],
        label="AI threshold strategy"
    )
    plt.plot(
        bt.index, bt["buy_hold_equity"],
        label="Buy & hold"
    )
    plt.title(f"{ticker} — unseen test-period equity curves")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📊 Recent model observations")
    show_cols = [
        "Close", "rsi_14", "ret_5", "ret_20",
        "vol_20", "prob_up", "signal"
    ]
    st.dataframe(
        bt[show_cols].tail(20),
        use_container_width=True
    )

    st.subheader("📥 Export research results")
    csv = bt.to_csv().encode("utf-8")
    st.download_button(
        "Download backtest CSV",
        data=csv,
        file_name=f"{ticker}_ai_backtest.csv",
        mime="text/csv",
    )

    st.subheader("What this model still does NOT know")
    st.markdown("""
- It does not yet use company fundamentals.
- It does not yet understand news or macroeconomic events.
- It does not yet detect sophisticated market regimes.
- It does not model every execution/friction issue.
- It does not guarantee that historical relationships persist.
- It does not place trades.

**Next research versions:** walk-forward validation → multi-asset scanning →
market-regime detection → news/fundamental features → paper trading.
""")

except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    st.info(
        "Check the ticker and internet connection. For the first run, "
        "install the requirements and try again."
    )
