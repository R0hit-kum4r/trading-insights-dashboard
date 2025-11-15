"""
Trading Insights Dashboard (final, single-file)

Requirements:
    pip install streamlit yfinance pandas numpy plotly

Run:
    streamlit run Trading.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from io import BytesIO

st.set_page_config(layout="wide", page_title="Trading Insights Dashboard (Final)")

# ---------- Utility indicator functions ----------
def sma(series, window):
    return series.rolling(window=window).mean()

def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

# ---------- Simple trend predictor using numpy ----------
def simple_trend_predict(close_series, days_forward=7, lookback=60):
    series = close_series.dropna().copy()
    if series.empty:
        return None
    lookback = min(lookback, len(series))
    if lookback < 5:
        return None
    x = np.arange(lookback)
    y = np.log(series.values[-lookback:])
    try:
        coeffs = np.polyfit(x, y, 1)
    except Exception:
        return None
    slope, intercept = coeffs[0], coeffs[1]
    future_x = np.arange(lookback, lookback + days_forward)
    pred_log = slope * future_x + intercept
    pred = np.exp(pred_log)
    pred_index = pd.date_range(start=series.index[-1] + pd.Timedelta(1, unit='D'),
                               periods=days_forward, freq='B')
    return pd.Series(pred, index=pred_index)

# ---------- Helpers ----------
def df_to_csv_bytes(df):
    b = BytesIO()
    df.to_csv(b, index=True)
    b.seek(0)
    return b.getvalue()

# ---------- Sidebar (with Indian tickers quick select) ----------
st.sidebar.header("Inputs")
input_mode = st.sidebar.radio("Data source", ("Ticker (Yahoo Finance)", "Upload CSV"))

# Popular Indian tickers dropdown (optional quick select)
indian_examples = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","TATAMOTORS.NS","LT.NS"]
if input_mode == "Ticker (Yahoo Finance)":
    ticker_input = st.sidebar.text_input("Ticker (e.g. AAPL, BTC-USD, RELIANCE.NS)", value="AAPL")
    use_example = st.sidebar.selectbox("Or pick example (India)", ["— none —"] + indian_examples, index=0)
    if use_example != "— none —":
        ticker_input = use_example
    ticker = ticker_input.strip().upper()
    period = st.sidebar.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=0)
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    fetch_button = st.sidebar.button("Fetch data")
else:
    upload = st.sidebar.file_uploader("Upload CSV (must have Date and Close columns)", type=['csv'])

prediction_days = st.sidebar.slider("Prediction horizon (business days)", 1, 30, 7)
lookback = st.sidebar.slider("Trend lookback window (days)", 20, 365, 90)
show_indicators = st.sidebar.multiselect("Indicators", ["SMA", "EMA", "MACD", "RSI"], default=["SMA", "MACD"])

# Candles + Bollinger options
show_candles = st.sidebar.checkbox("Candlestick", value=True)
show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=True)

# Bollinger sliders (appear when BB enabled)
if show_bollinger:
    bb_window = st.sidebar.slider("BB Window (period)", 10, 50, 20)
    bb_std = st.sidebar.slider("BB Std Dev (volatility)", 1.0, 4.0, 2.0)
else:
    bb_window = 20
    bb_std = 2.0

st.sidebar.markdown("---")
st.sidebar.markdown("Built for: Traders (sell via Telegram/YouTube community)")

# ---------- Data loading with caching ----------
@st.cache_data
def load_yf(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        # flatten columns if multiindex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(x) for x in col]).strip() for col in df.columns.values]
        df.index = pd.to_datetime(df.index, errors='coerce')
        return df
    except Exception:
        return None

@st.cache_data
def load_csv(file):
    try:
        df = pd.read_csv(file, parse_dates=True, infer_datetime_format=True)
        # common date column names
        if 'Date' in df.columns:
            df = df.set_index('Date')
        elif 'date' in df.columns:
            df = df.set_index('date')
        # flatten columns if multiindex-like (unlikely for CSV)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join([str(x) for x in col]).strip() for col in df.columns.values]
        df.index = pd.to_datetime(df.index, errors='coerce')
        return df
    except Exception:
        return None

# ---------- Fetch data (button-controlled) ----------
data = None
if input_mode == "Ticker (Yahoo Finance)":
    if fetch_button:
        data = load_yf(ticker, period, interval)
        if data is None:
            st.error("Failed to fetch data from Yahoo Finance. Check ticker, internet, or try another interval/period.")
        else:
            st.success(f"Fetched {len(data)} rows for {ticker}")
else:
    if upload is not None:
        data = load_csv(upload)
        if data is None:
            st.error("Failed to load CSV. Make sure it has 'Date' (or 'date') and 'Close' columns.")
        else:
            st.success(f"Loaded CSV with {len(data)} rows")

if data is None:
    st.info("Waiting for data. Provide a ticker and click 'Fetch data' or upload a CSV.")
    st.stop()

# ---------- Prepare DataFrame ----------
df = data.copy()

# ---------- IMPROVED FIX: robust coercion & MultiIndex handling ----------
# If columns are MultiIndex already flattened in loader, else flatten now
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join([str(x) for x in col]).strip() for col in df.columns.values]

# Try common names directly first
possible_close_names = ['Close', 'close', 'Adj Close', 'Adj_Close', 'adjclose', 'Close*', 'CLOSE', 'AdjClose']
close_col = None
for name in possible_close_names:
    if name in df.columns:
        close_col = df[name]
        break

# If still not found, try substring match
if close_col is None:
    for c in df.columns:
        if 'close' in str(c).lower():
            close_col = df[c]
            break

# Heuristic: numeric columns
if close_col is None:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        close_col = df[numeric_cols[0]]
    elif len(numeric_cols) > 1:
        chosen = None
        for c in numeric_cols:
            if 'close' in str(c).lower():
                chosen = c
                break
        close_col = df[chosen] if chosen is not None else df[numeric_cols[-1]]

# If df is a single-row where values might be the series
if close_col is None:
    try:
        if df.shape[0] == 1 and df.shape[1] >= 1:
            tmp = pd.Series(df.iloc[0].values, index=df.columns)
            # try to find close-like in this tmp
            for c in tmp.index:
                if 'close' in str(c).lower():
                    close_col = tmp[c]
                    break
            if close_col is None:
                # fallback: numeric values
                numeric_vals = [v for v in tmp.values if isinstance(v, (int, float, np.floating, np.integer))]
                if numeric_vals:
                    close_col = pd.Series(numeric_vals)
    except Exception:
        close_col = None

# If close_col is array/list/tuple -> convert to Series with index when possible
if isinstance(close_col, (np.ndarray, list, tuple)):
    try:
        if len(close_col) == len(df.index):
            close_col = pd.Series(close_col, index=df.index)
        else:
            close_col = pd.Series(close_col)
    except Exception:
        pass

# Final coercion to numeric
if close_col is None:
    st.error("No Close-like column found in the data. Try a different ticker, interval, or upload a CSV with Close column.")
    st.write("DEBUG: columns =", list(df.columns))
    st.stop()

try:
    df['Close'] = pd.to_numeric(close_col, errors='coerce')
except Exception:
    try:
        df['Close'] = pd.Series(close_col).astype(float)
    except Exception:
        st.error("Failed to interpret Close values. Try different ticker/interval or upload a CSV with Close column.")
        st.stop()

# If still empty, provide helpful debug and stop
if df['Close'].dropna().empty:
    st.error("No valid numeric Close prices found for this ticker. Try interval = '1wk' or '1mo', or a different ticker.")
    st.write("DEBUG: columns =", list(df.columns))
    try:
        st.write(df.head())
    except Exception:
        pass
    st.stop()

# ---------- Compute indicators ----------
if 'SMA' in show_indicators:
    df['SMA_20'] = sma(df['Close'], 20)
    df['SMA_50'] = sma(df['Close'], 50)
if 'EMA' in show_indicators:
    df['EMA_20'] = ema(df['Close'], 20)
if 'MACD' in show_indicators:
    m_line, m_signal, m_hist = macd(df['Close'])
    df['MACD_line'] = m_line
    df['MACD_signal'] = m_signal
    df['MACD_hist'] = m_hist
if 'RSI' in show_indicators:
    df['RSI_14'] = rsi(df['Close'], 14)

# ---------- Prediction ----------
pred_series = simple_trend_predict(df['Close'], days_forward=prediction_days, lookback=lookback)

# ---------- Layout / Charts ----------
st.title("Trading Insights Dashboard (Final)")

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Price & Prediction")

    # Ensure OHLC columns exist (try common variants and also try flattened names)
    ohlc_names = {
        'open': ['Open', 'open', 'OPEN', 'Open_'],
        'high': ['High', 'high', 'HIGH', 'High_'],
        'low':  ['Low', 'low', 'LOW', 'Low_'],
        'close':['Close', 'close', 'CLOSE', 'Adj Close', 'Adj_Close', 'Close_']
    }

    # helper to find a column by possible names or by substring
    def find_col(possible_names):
        for n in possible_names:
            if n in df.columns:
                return n
        # substring match
        for c in df.columns:
            for n in possible_names:
                if n.lower().strip('_') in str(c).lower():
                    return c
        return None

    col_open = find_col(ohlc_names['open'])
    col_high = find_col(ohlc_names['high'])
    col_low  = find_col(ohlc_names['low'])
    col_close= find_col(ohlc_names['close'])

    # coerce numeric where found
    for c in [col_open, col_high, col_low, col_close]:
        if c is not None and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # If we don't have OHLC, fallback: use Close only (candlestick won't show)
    has_ohlc = all([col_open in df.columns if col_open is not None else False,
                    col_high in df.columns if col_high is not None else False,
                    col_low in df.columns if col_low is not None else False,
                    col_close in df.columns if col_close is not None else False])

    # Compute Bollinger Bands using sliders
    if show_bollinger and 'Close' in df.columns:
        df['BB_MA'] = df['Close'].rolling(window=bb_window).mean()
        df['BB_STD'] = df['Close'].rolling(window=bb_window).std()
        df['BB_UPPER'] = df['BB_MA'] + bb_std * df['BB_STD']
        df['BB_LOWER'] = df['BB_MA'] - bb_std * df['BB_STD']

    # Build plotly figure
    fig = go.Figure()

    # If user asked for candlesticks and we have OHLC, draw candlestick safely
    if show_candles and has_ohlc:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df[col_open],
            high=df[col_high],
            low=df[col_low],
            close=df[col_close],
            name='OHLC',
            increasing_line_color='green',
            decreasing_line_color='red',
            showlegend=True
        ))
    else:
        # fallback: show Close line so chart isn't empty
        if show_candles and not has_ohlc:
            # Helpful tip message for user
            st.warning("OHLC data not available for this interval/ticker. Showing Close line instead. For daily candlesticks use interval = '1d'.")
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', mode='lines', line=dict(width=2)))

    # Add SMA/EMA lines if present and requested
    if 'SMA_20' in df.columns and 'SMA' in show_indicators:
        if not df['SMA_20'].dropna().empty:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', mode='lines'))
    if 'EMA_20' in df.columns and 'EMA' in show_indicators:
        if not df['EMA_20'].dropna().empty:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', mode='lines'))

    # Add Bollinger Bands
    if show_bollinger and 'BB_UPPER' in df.columns:
        if not df['BB_UPPER'].dropna().empty:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], name='BB Upper', mode='lines', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_MA'], name='BB Mid', mode='lines', opacity=0.6))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], name='BB Lower', mode='lines', line=dict(dash='dash')))
            # fill between upper and lower for visual band
            fig.add_trace(go.Scatter(
                x=list(df.index) + list(df.index[::-1]),
                y=list(df['BB_UPPER']) + list(df['BB_LOWER'][::-1]),
                fill='toself',
                fillcolor='rgba(173,216,230,0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name='BB Band'
            ))

    # Prediction trace (if available)
    if pred_series is not None and not pred_series.dropna().empty:
        pred_vals = pd.to_numeric(pred_series.values, errors='coerce')
        if np.isfinite(pred_vals).any():
            try:
                pred_index = pd.to_datetime(pred_series.index)
            except Exception:
                pred_index = pred_series.index
            fig.add_trace(go.Scatter(x=pred_index, y=pred_vals, name=f'Prediction (+{prediction_days}d)',
                                     mode='lines', line=dict(dash='dash', width=2)))

    # Layout tweaks
    fig.update_layout(
        height=560,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.01),
        margin=dict(l=40, r=20, t=50, b=40)
    )

    # Finally render
    if len(fig.data) == 0:
        st.error("No traces to display in the chart.")
    else:
        st.plotly_chart(fig, use_container_width=True)

    # MACD subplot
    if 'MACD_line' in df.columns and 'MACD_signal' in df.columns and not df['MACD_line'].dropna().empty:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_line'], name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal'))
        fig_macd.update_layout(height=240, template='plotly_dark', margin=dict(l=40, r=20, t=20, b=20))
        st.plotly_chart(fig_macd, use_container_width=True)

    # RSI small chart
    if 'RSI_14' in df.columns and not df['RSI_14'].dropna().empty:
        st.subheader("RSI")
        st.line_chart(df['RSI_14'].dropna())

with col2:
    st.subheader("Quick Metrics")
    close_non_na = df['Close'].dropna()
    if close_non_na.empty:
        st.info("No close prices available to show metrics.")
    else:
        latest_close = float(close_non_na.iloc[-1])
        st.metric("Latest Close", f"{latest_close:.2f}")

        if pred_series is not None and len(pred_series.dropna()) > 0:
            predicted_last = float(pd.to_numeric(pred_series.dropna().iloc[-1]))
            st.metric(f"Predicted Close (+{prediction_days}d)", f"{predicted_last:.2f}")
            pct = (predicted_last / latest_close - 1) * 100
            st.metric("Predicted % change", f"{pct:.2f}%")
        else:
            st.info("Prediction not available (not enough data).")

    st.markdown("---")
    st.download_button("Download data (CSV)", data=df_to_csv_bytes(df), file_name='historical_with_indicators.csv')

# ---------- Simple backtest ----------
st.subheader("Simple Backtest: SMA 20/50 Crossover")
if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
    df_bt = df.dropna().copy()
    if len(df_bt) < 2:
        st.info("Not enough data to run backtest after dropping NaNs.")
    else:
        df_bt['signal'] = 0
        df_bt.loc[df_bt['SMA_20'] > df_bt['SMA_50'], 'signal'] = 1
        df_bt['position'] = df_bt['signal'].diff().fillna(0)
        df_bt['ret'] = df_bt['Close'].pct_change().fillna(0)
        df_bt['strat_ret'] = df_bt['ret'] * df_bt['signal'].shift(1).fillna(0)
        cum = (1 + df_bt['strat_ret']).cumprod()
        buy_and_hold = (1 + df_bt['ret']).cumprod()
        st.line_chart(pd.DataFrame({'Strategy': cum, 'Buy & Hold': buy_and_hold}))
        st.write("Total strategy return:", f"{(cum.iloc[-1] - 1) * 100:.2f}%")
        st.write("Buy & Hold return:", f"{(buy_and_hold.iloc[-1] - 1) * 100:.2f}%")
else:
    st.info("Enable SMA in sidebar to run the example backtest (SMA 20 & SMA 50 required).")

# ---------- Notes ----------
st.markdown("---")
st.header("How to present/sell this to traders")
st.markdown(
    "- Short video clips showing how to fetch a ticker, read prediction, and interpret MACD/RSI.\n"
    "- Weekly PDF with top 5 tickers (based on predicted % change) for subscribers.\n"
    "- Downloadable CSV for importing into Telegram bots or spreadsheets."
)
st.caption("Made with Streamlit • Simple trend prediction is a statistical extrapolation and not financial advice.")
