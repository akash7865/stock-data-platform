

import yfinance as yf
import pandas as pd
import sqlite3
import os


# CONFIGURATION

STOCKS = {
    "RELIANCE":   "RELIANCE.NS",
    "TCS":        "TCS.NS",
    "INFY":       "INFY.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "ICICIBANK":  "ICICIBANK.NS",
    "WIPRO":      "WIPRO.NS",
    "SBIN":       "SBIN.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "ADANIENT":   "ADANIENT.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
}

DB_PATH  = os.path.join(os.path.dirname(__file__), "database.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "stock_data.csv")


# STEP 1: FETCH

def fetch_stock_data(period: str = "1y") -> pd.DataFrame:
    """
    Download historical OHLCV data for all configured stocks.
    Skips any ticker that returns empty data or raises an error.
    """
    print(f"📡 Fetching data for {len(STOCKS)} NSE stocks...")
    all_data = []

    for name, ticker in STOCKS.items():
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

            if df.empty:
                print(f"  ⚠️  No data returned for {name} ({ticker}) — skipping.")
                continue

            # Flatten MultiIndex columns produced by yfinance ≥0.2.x
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [str(c).strip() for c in df.columns]
            df["symbol"] = name
            df["ticker"] = ticker
            all_data.append(df)
            print(f"  ✅ {name}: {len(df)} rows")

        except Exception as exc:
            print(f"  ❌ Failed to fetch {name}: {exc}")

    if not all_data:
        raise RuntimeError(
            "No stock data fetched. Check your internet connection or ticker symbols."
        )

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n📦 Total rows fetched: {len(combined)}")
    return combined


# STEP 2: CLEAN

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names, coerce types safely, drop bad rows.

    Key safety improvements over a naive approach:
    - pd.to_datetime(..., errors='coerce') turns unparseable dates into NaT
      instead of raising an exception, so one bad row cannot crash the pipeline.
    - pd.to_numeric(..., errors='coerce') converts non-numeric cells to NaN
      rather than raising ValueError.
    - Both are then caught by dropna() so no corrupted rows survive to the DB.
    """
    print("\n🧹 Cleaning data...")
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Handle yfinance index column naming edge cases
    if "date" not in df.columns and "index" in df.columns:
        df.rename(columns={"index": "date"}, inplace=True)

    # Coerce dates — bad values become NaT (caught by dropna below)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    required = ["open", "high", "low", "close", "volume"]

    # Coerce numerics — non-numeric cells become NaN
    for col in required:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaT date or NaN in any required OHLCV column
    df.dropna(subset=["date"] + required, inplace=True)

    # Drop duplicate (symbol, date) pairs — keep the last occurrence
    df.drop_duplicates(subset=["symbol", "date"], keep="last", inplace=True)

    df.sort_values(["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  ✅ {len(df)} rows | {df['symbol'].nunique()} companies")
    return df


# STEP 3: ENRICH WITH METRICS

def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-stock financial metrics.

    daily_return (%)
        = (close - open) / open × 100
        Convention: stored as a percentage.  +1.5 means the stock gained 1.5%.
        Guard: if open == 0 the result is NaN — no divide-by-zero crash.

    ma_7, ma_30
        Rolling mean of close price over 7 / 30 trading days.

    high_52w, low_52w
        Rolling max / min of close over 252 trading days (≈ 1 calendar year).

    volatility_30d  [custom metric]
        30-day rolling standard deviation of daily_return (%).
        Higher value → the stock has been swinging more recently.

    price_momentum  [custom metric]
        Percentage price change versus 30 trading days ago.
        = (close - close_30d_ago) / close_30d_ago × 100
        Guard: close_30d_ago == 0 is replaced with NaN.
    """
    print("\n📐 Calculating metrics...")
    parts = []

    for symbol, grp in df.groupby("symbol"):
        g = grp.copy().sort_values("date").reset_index(drop=True)

        # daily_return (%) — guard open == 0 to avoid ZeroDivisionError
        safe_open = g["open"].replace(0, float("nan"))
        g["daily_return"] = ((g["close"] - safe_open) / safe_open * 100).round(4)

        # Moving averages of close price
        g["ma_7"]  = g["close"].rolling(7,   min_periods=1).mean().round(2)
        g["ma_30"] = g["close"].rolling(30,  min_periods=1).mean().round(2)

        # 52-week rolling high / low (252 trading days ≈ 1 year)
        g["high_52w"] = g["close"].rolling(252, min_periods=1).max().round(2)
        g["low_52w"]  = g["close"].rolling(252, min_periods=1).min().round(2)

        # Volatility: 30-day rolling std of daily_return (%)
        g["volatility_30d"] = (
            g["daily_return"].rolling(30, min_periods=5).std().round(4)
        )

        # Price momentum: % change vs 30 trading days ago
        prev_30 = g["close"].shift(30).replace(0, float("nan"))
        g["price_momentum"] = ((g["close"] - prev_30) / prev_30 * 100).round(4)

        parts.append(g)

    result = pd.concat(parts, ignore_index=True)
    print(
        "  ✅ Metrics: daily_return (%), ma_7, ma_30, "
        "high_52w, low_52w, volatility_30d (%), price_momentum (%)"
    )
    return result


# STEP 4: SAVE


def save_to_csv(df: pd.DataFrame) -> None:
    """Write the enriched dataframe to CSV."""
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n💾 CSV  → {CSV_PATH}")


def save_to_sqlite(df: pd.DataFrame) -> None:
    """
    Persist data to SQLite using a context manager (auto-commit, auto-close).

    Two tables:
      stock_data      Full daily history for every symbol with all metrics.
      stock_snapshot  One row per symbol: the most recent trading day.
                      Used by fast summary and gainers/losers queries.

    Note: sqlite3 is part of the Python standard library — no pip install needed.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    df_db = df.copy()
    df_db["date"] = df_db["date"].astype(str)   # SQLite stores dates as TEXT

    # Latest row per symbol for the snapshot table
    snapshot = (
        df_db.sort_values("date")
             .groupby("symbol", as_index=False)
             .last()
    )

    with sqlite3.connect(DB_PATH) as conn:
        df_db.to_sql("stock_data",     conn, if_exists="replace", index=False)
        snapshot.to_sql("stock_snapshot", conn, if_exists="replace", index=False)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_date "
            "ON stock_data(symbol, date);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshot_symbol "
            "ON stock_snapshot(symbol);"
        )
        # Context manager calls conn.commit() automatically on exit

    print(f"💾 DB   → {DB_PATH}  (tables: stock_data, stock_snapshot)")


# PIPELINE ENTRY POINT

def run_pipeline() -> pd.DataFrame:
    """Fetch → Clean → Enrich → Save. Returns the enriched dataframe.
    If live Yahoo fetch fails, fall back to bundled CSV data.
    """
    print("=" * 55)
    print("  🚀 Stock Data Pipeline")
    print("=" * 55)

    try:
        raw = fetch_stock_data(period="1y")
        cleaned = clean_data(raw)
        enriched = add_metrics(cleaned)
        save_to_csv(enriched)
        save_to_sqlite(enriched)

        print("\n" + "=" * 55)
        print("  ✅ Pipeline complete (live Yahoo data)")
        print(f"  📊 {enriched['symbol'].nunique()} stocks | {len(enriched):,} rows")
        print(f"  📅 {enriched['date'].min().date()} → {enriched['date'].max().date()}")
        print("=" * 55)
        return enriched

    except Exception as exc:
        print(f"\n⚠️ Live fetch failed: {exc}")
        print("📦 Trying fallback CSV...")

        fallback_csv = os.path.join(os.path.dirname(__file__), "stock_data.csv")
        if not os.path.exists(fallback_csv):
            raise RuntimeError(
                "Live fetch failed and fallback stock_data.csv was not found."
            )

        df = pd.read_csv(fallback_csv)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)

        save_to_sqlite(df)

        print("\n" + "=" * 55)
        print("  ✅ Pipeline complete (fallback CSV)")
        print(f"  📊 {df['symbol'].nunique()} stocks | {len(df):,} rows")
        print(f"  📅 {df['date'].min().date()} → {df['date'].max().date()}")
        print("=" * 55)
        return df
