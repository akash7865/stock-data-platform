"""
utils.py — Database Helpers, ML Prediction, Market Analytics
──────────────────────────────────────────────────────────────
All values returned use the same percentage convention as data.py:
  daily_return, volatility_30d, price_momentum are stored as %.
  e.g. daily_return = +1.25 means +1.25% gain.
"""

import sqlite3
import os
import pandas as pd
import numpy as np
from datetime import timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")


# ─────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Return a live SQLite connection, or raise FileNotFoundError with a clear message."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            "database.db not found. Run the pipeline first:  python backend/data.py"
        )
    return sqlite3.connect(DB_PATH)


def get_all_symbols() -> list[str]:
    """Return sorted list of all stock symbols present in the database."""
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT DISTINCT symbol FROM stock_data ORDER BY symbol", conn
        )
    return df["symbol"].tolist()


def symbol_exists(symbol: str) -> bool:
    """Return True if the symbol has rows in stock_data."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM stock_data WHERE symbol = ? LIMIT 1", (symbol.upper(),)
        ).fetchone()
    return row is not None


def get_stock_df(symbol: str, days: int = 365) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql(
            """
            SELECT * FROM stock_data
            WHERE  symbol = ?
            ORDER  BY date DESC
            LIMIT  ?
            """,
            conn,
            params=(symbol.upper(), min(days, 300)),  # ← change this line
        )

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.fillna(0)
    return df


# ─────────────────────────────────────────────
# ML PRICE PREDICTION  (Linear Regression)
# ─────────────────────────────────────────────

def predict_prices(symbol: str, days_ahead: int = 7) -> dict:
    """
    Trend-based demo prediction using Linear Regression.

    How it works:
      - Uses the last 90 trading days of close prices.
      - Encodes each row as an integer day index (0, 1, 2, …).
      - Fits a straight line through those points.
      - Extrapolates N steps ahead.

    Important caveats returned in the response:
      - R² measures how well a straight line fits past data, NOT future accuracy.
      - If R² < 0.5 a low-confidence warning is included.
      - This model cannot capture volatility, news events, or market regime changes.
    """
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return {"error": "scikit-learn not installed.  Run: pip install scikit-learn"}

    df = get_stock_df(symbol, days=90)
    if len(df) < 20:
        return {"error": f"Need at least 20 data points for {symbol}. Got {len(df)}."}

    df = df.reset_index(drop=True)
    df["day_index"] = np.arange(len(df))

    X = df[["day_index"]].values
    y = df["close"].values

    model = LinearRegression()
    model.fit(X, y)

    last_idx = int(df["day_index"].iloc[-1])
    future_X = np.arange(last_idx + 1, last_idx + 1 + days_ahead).reshape(-1, 1)
    predicted = model.predict(future_X).round(2)

    # Future trading dates (skip weekends)
    last_date   = df["date"].iloc[-1]
    future_dates = []
    current      = last_date
    while len(future_dates) < days_ahead:
        current += timedelta(days=1)
        if current.weekday() < 5:
            future_dates.append(current.strftime("%Y-%m-%d"))

    r2 = round(float(model.score(X, y)), 4)
    low_confidence = r2 < 0.5
    warning = (
        "⚠️ Low R² score — the historical price trend is not strongly linear. "
        "Treat this prediction with extra caution."
        if low_confidence else None
    )

    return {
        "symbol":  symbol,
        "model":   "Linear Regression (trend-based demo prediction)",
        "r2_score": r2,
        "r2_note": (
            "R² measures how well a straight line fits PAST data. "
            "It does NOT measure future forecast accuracy."
        ),
        "warning": warning,
        "disclaimer": "Not financial advice. For educational demonstration only.",
        "historical": {
            "dates":  df["date"].dt.strftime("%Y-%m-%d").tolist()[-30:],
            "prices": df["close"].round(2).tolist()[-30:],
        },
        "prediction": {
            "dates":  future_dates,
            "prices": predicted.tolist(),
        },
    }


# ─────────────────────────────────────────────
# TOP GAINERS & LOSERS
# ─────────────────────────────────────────────

def get_top_gainers_losers(n: int = 5) -> dict:
    """
    Return the top-N gainers and top-N losers for the most recent trading day.
    Uses the stock_snapshot table (one row per symbol = latest day) for speed.
    n is clamped to the number of available symbols to avoid empty slices.
    """
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT symbol, close, daily_return FROM stock_snapshot", conn
        )
        latest_date = pd.read_sql(
            "SELECT MAX(date) AS d FROM stock_snapshot", conn
        )["d"].iloc[0]

    if df.empty:
        return {"error": "No snapshot data available. Re-run the pipeline."}

    # Clamp n so we never request more rows than exist
    n = min(n, len(df) // 2 or 1)

    df["close"]        = df["close"].round(2)
    df["daily_return"] = df["daily_return"].round(4)
    df.sort_values("daily_return", ascending=False, inplace=True)

    gainers = df.head(n)[["symbol", "close", "daily_return"]].to_dict(orient="records")
    losers  = df.tail(n)[["symbol", "close", "daily_return"]].iloc[::-1].to_dict(orient="records")

    return {
        "date":        latest_date,
        "top_gainers": gainers,
        "top_losers":  losers,
    }


# ─────────────────────────────────────────────
# CORRELATION  (Custom metric)
# ─────────────────────────────────────────────

def get_correlation(symbol1: str, symbol2: str, days: int = 90) -> dict:
    """
    Pearson correlation of daily returns between two stocks.

    Guards:
      - Both symbols are validated before any DB query.
      - Comparing a stock to itself returns a friendly message.
      - Fewer than 10 overlapping rows returns an error instead of crashing.

    Interpretation:
       >= 0.7   Strong positive  — move together
       0.3–0.7  Moderate positive
      -0.3–0.3  Little / no relationship
      -0.7–-0.3 Moderate negative
      <= -0.7   Strong negative  — move opposite
    """
    s1, s2 = symbol1.upper(), symbol2.upper()

    if s1 == s2:
        return {
            "error": (
                f"Both symbols are '{s1}'. "
                "Please select two different stocks to compare."
            )
        }

    for sym in (s1, s2):
        if not symbol_exists(sym):
            return {"error": f"Symbol '{sym}' not found in the database."}

    df1 = get_stock_df(s1, days=days)[["date", "daily_return"]].rename(
        columns={"daily_return": s1}
    )
    df2 = get_stock_df(s2, days=days)[["date", "daily_return"]].rename(
        columns={"daily_return": s2}
    )

    merged = pd.merge(df1, df2, on="date", how="inner").dropna()

    if len(merged) < 10:
        return {
            "error": (
                f"Only {len(merged)} overlapping trading days found. "
                "Need at least 10 to compute a meaningful correlation."
            )
        }

    corr = round(float(merged[s1].corr(merged[s2])), 4)

    if   corr >=  0.7:  interp = "Strong positive — stocks tend to move together."
    elif corr >=  0.3:  interp = "Moderate positive correlation."
    elif corr >= -0.3:  interp = "Little to no correlation."
    elif corr >= -0.7:  interp = "Moderate negative correlation."
    else:               interp = "Strong negative — stocks tend to move opposite."

    return {
        "symbol1":        s1,
        "symbol2":        s2,
        "correlation":    corr,
        "interpretation": interp,
        "data_points":    len(merged),
        "period_days":    days,
    }

