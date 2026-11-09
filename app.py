"""Streamlit dashboard: LSTM stock price trend prediction.

Run locally:  streamlit run app.py
Deploy:       push this folder to GitHub, then create the app at share.streamlit.io
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

st.set_page_config(page_title="LSTM Stock Trend Predictor", layout="wide")
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["LogRet", "MA20_dev", "MA50_dev", "RSI14_n"]


# ---------- data ----------
@st.cache_data(show_spinner=False)
def load_prices(ticker, start):
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False, timeout=15)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) < 300:
            raise ValueError("too few rows")
        return df, "Yahoo Finance (live)"
    except Exception:
        path = os.path.join(HERE, "data", "aapl_fallback.csv")
        if ticker.upper() == "AAPL" and os.path.exists(path):
            raw = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
            raw = raw[raw["Date"] >= "2000-01-01"].set_index("Date")
            df = raw[["Open", "High", "Low", "Adj Close", "Volume"]].rename(columns={"Adj Close": "Close"})
            return df, "Bundled AAPL file 2000-2008 (Yahoo unreachable)"
        return None, "unavailable"


def rsi_wilder(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)


def add_indicators(df):
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI14"] = rsi_wilder(df["Close"])
    df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))
    df["MA20_dev"] = df["Close"] / df["MA20"] - 1
    df["MA50_dev"] = df["Close"] / df["MA50"] - 1
    df["RSI14_n"] = df["RSI14"] / 100
    return df.dropna()


def sequences(X, y, window, start, stop):
    idx = list(range(max(start, window), stop))
    return np.array([X[i - window:i] for i in idx]), y[idx], np.array(idx)


# ---------- model ----------
@st.cache_resource(show_spinner=False)
def train(ticker, start, window, epochs):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(42)
    np.random.seed(42)
    df, source = load_prices(ticker, start)
    df = add_indicators(df)
    split = int(len(df) * 0.8)
    xs = StandardScaler().fit(df[FEATURES].iloc[:split])
    ys = StandardScaler().fit(df[["LogRet"]].iloc[:split])
    X, y = xs.transform(df[FEATURES]), ys.transform(df[["LogRet"]]).ravel()
    X_tr, y_tr, _ = sequences(X, y, window, 0, split)
    X_te, _, i_te = sequences(X, y, window, split, len(df))

    model = keras.Sequential([
        layers.Input(shape=(window, len(FEATURES))),
        layers.LSTM(64, return_sequences=True), layers.Dropout(0.2),
        layers.LSTM(32), layers.Dropout(0.2),
        layers.Dense(16, activation="relu"), layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    nv = int(len(X_tr) * 0.1)
    hist = model.fit(
        X_tr[:-nv], y_tr[:-nv], validation_data=(X_tr[-nv:], y_tr[-nv:]),
        epochs=epochs, batch_size=32, shuffle=False, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])

    prev = df["Close"].values[i_te - 1]
    pred = prev * np.exp(ys.inverse_transform(model.predict(X_te, verbose=0)).ravel())
    actual = df["Close"].values[i_te]
    last = X[-window:].reshape(1, window, len(FEATURES))
    nxt = float(df["Close"].iloc[-1] * np.exp(ys.inverse_transform(model.predict(last, verbose=0))[0, 0]))
    return dict(df=df, source=source, hist=hist.history, dates=df.index[i_te],
                actual=actual, pred=pred, prev=prev, next_close=nxt)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------- UI ----------
st.title("Stock Price Trend Prediction with LSTM")
st.caption("Educational project. Not investment advice.")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    start = st.date_input("History start", pd.Timestamp("2015-01-01")).strftime("%Y-%m-%d")
    window = st.slider("Look-back window (days)", 20, 120, 60, 10)
    epochs = st.slider("Max epochs", 5, 60, 30, 5)
    run = st.button("Fetch data and train", type="primary")

if not run:
    st.info("Pick a ticker in the sidebar and press **Fetch data and train**.")
    st.stop()

df0, _ = load_prices(ticker, start)
if df0 is None:
    st.error(f"Could not fetch data for {ticker}. Yahoo Finance may be unreachable from this network.")
    st.stop()

with st.spinner("Training the LSTM (about a minute on CPU)..."):
    r = train(ticker, start, window, epochs)

df = r["df"]
st.write(f"**Data source:** {r['source']}  |  {df.index.min().date()} to {df.index.max().date()}  |  {len(df):,} rows")

c1, c2, c3 = st.columns(3)
c1.metric("Last close", f"{df['Close'].iloc[-1]:.2f}")
c2.metric("LSTM next-day forecast", f"{r['next_close']:.2f}", f"{(r['next_close'] / df['Close'].iloc[-1] - 1) * 100:+.2f}%")
c3.metric("Test RMSE: LSTM vs naive", f"{rmse(r['actual'], r['pred']):.3f}", f"naive {rmse(r['actual'], r['prev']):.3f}", delta_color="off")
up = np.mean(np.sign(r["actual"] - r["prev"]) == np.sign(r["pred"] - r["prev"])) * 100
st.caption(f"Directional accuracy on the test period: {up:.1f}% (share of up-days: {np.mean(r['actual'] > r['prev']) * 100:.1f}%). "
           "If the LSTM RMSE is close to the naive RMSE, the model is mostly echoing yesterday's close.")

t1, t2, t3 = st.tabs(["Predictions vs actual", "Price, MA and RSI", "Training loss"])
with t1:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(r["dates"], r["actual"], label="Actual")
    ax.plot(r["dates"], r["pred"], label="LSTM prediction")
    ax.set_ylabel("Price"); ax.legend()
    st.pyplot(fig)
with t2:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(df.index, df["Close"], label="Close"); a1.plot(df.index, df["MA20"], label="MA20"); a1.plot(df.index, df["MA50"], label="MA50")
    a1.legend()
    a2.plot(df.index, df["RSI14"], color="purple"); a2.axhline(70, color="red", ls="--"); a2.axhline(30, color="green", ls="--"); a2.set_ylim(0, 100)
    st.pyplot(fig)
with t3:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r["hist"]["loss"], label="train"); ax.plot(r["hist"]["val_loss"], label="validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (scaled)"); ax.legend()
    st.pyplot(fig)
