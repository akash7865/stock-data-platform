"""
main.py — FastAPI Backend for Stock Data Intelligence Dashboard
────────────────────────────────────────────────────────────────
Run:       uvicorn main:app --reload          (from backend/ folder)
Swagger:   http://localhost:8000/docs
Dashboard: http://localhost:8000/dashboard
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import os

from utils import (
    get_all_symbols,
    get_stock_df,
    predict_prices,
    get_top_gainers_losers,
    get_correlation,
    symbol_exists,
)
from ml_models import predict_lstm, analyze_sentiment, predict_all_stocks

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

APP_VERSION = "1.1.0"

app = FastAPI(
    title="📈 Stock Data Intelligence Dashboard",
    description="""
A mini financial data platform built with **FastAPI**, **yfinance**, and **scikit-learn / TensorFlow**.

## Endpoints
| Tag | What it does |
|-----|-------------|
| **General** | Health check, root info, dashboard UI |
| **Stocks** | History, summary metrics per stock |
| **ML Prediction** | Linear regression trend prediction |
| **Market** | Gainers/losers, stock correlation |
| **Advanced ML** | LSTM deep learning, sentiment analysis, multi-stock ranking |

## Data
Real NSE data fetched from Yahoo Finance via `yfinance`.
`daily_return` and all percentage metrics are stored as **percent values**
(e.g. `+1.25` means the stock gained 1.25% that day).

> ⚠️ Educational platform only. Not financial advice.
    """,
    version=APP_VERSION,
)

# Allow the HTML/JS frontend to call the API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ─────────────────────────────────────────────
# GENERAL
# ─────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    """API info, version, and link to Swagger docs."""
    return {
        "app":       "Stock Data Intelligence Dashboard",
        "version":   APP_VERSION,
        "docs":      "/docs",
        "dashboard": "/dashboard",
        "health":    "/health",
        "endpoints": {
            "stocks":      ["/stocks", "/stocks/{symbol}", "/stocks/{symbol}/summary"],
            "prediction":  ["/stocks/{symbol}/predict", "/stocks/{symbol}/predict/lstm"],
            "market":      ["/market/gainers-losers", "/market/correlation", "/market/predict-all"],
            "sentiment":   ["/stocks/{symbol}/sentiment"],
        },
    }


@app.get("/health", tags=["General"])
def health_check():
    """
    Lightweight health check for Docker / deployment monitoring.
    Returns 200 OK when the server is up and the database is reachable.
    """
    db_ok = os.path.exists(
        os.path.join(os.path.dirname(__file__), "database.db")
    )
    if not db_ok:
        return JSONResponse(
            status_code=503,
            content={
                "status":  "degraded",
                "message": "database.db missing — run: python backend/data.py",
            },
        )
    return {"status": "ok", "version": APP_VERSION, "database": "reachable"}


@app.get("/dashboard", tags=["General"], include_in_schema=False)
def serve_dashboard():
    """Serve the frontend dashboard HTML page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        status_code=404,
        content={"message": "Frontend not found. Ensure frontend/index.html exists."},
    )


# ─────────────────────────────────────────────
# STOCKS
# ─────────────────────────────────────────────

@app.get("/stocks", tags=["Stocks"])
def list_stocks():
    """List all available stock symbols in the database."""
    try:
        symbols = get_all_symbols()
        return {"count": len(symbols), "symbols": symbols}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


@app.get("/stocks/{symbol}", tags=["Stocks"])
def get_stock_history(
    symbol: str,
    days: int = Query(default=90, ge=7, le=365, description="Trading days to return (7–365)"),
):
    """
    Historical OHLCV data + calculated metrics for a stock.

    - **symbol** e.g. `TCS`, `RELIANCE`, `INFY`
    - **days** — number of recent trading days (default 90)
    - **daily_return** is a percentage value (e.g. +1.25 = gained 1.25%)
    """
    try:
        df = get_stock_df(symbol.upper(), days=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data for '{symbol}'. Check available symbols at /stocks.",
        )

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return {
        "symbol":        symbol.upper(),
        "days_returned": len(df),
        "data":          df.to_dict(orient="records"),
    }


@app.get("/stocks/{symbol}/summary", tags=["Stocks"])
def get_stock_summary(symbol: str):
    """
    Latest key metrics for a stock:
    current price, daily return (%), MAs, 52W high/low, volatility, momentum.

    All percentage fields (daily_return_pct, volatility_score_30d,
    price_momentum_30d_pct) are in **percent** (e.g. +1.25 = +1.25%).
    """
    try:
        df = get_stock_df(symbol.upper(), days=365)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for '{symbol}'.")

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    def safe_float(val, default=0.0):
        try:
            v = float(val)
            return v if pd.notna(v) else default
        except (TypeError, ValueError):
            return default

    return {
        "symbol":                 symbol.upper(),
        "last_updated":           str(latest["date"].date()),
        "current_price":          round(safe_float(latest["close"]),    2),
        "open":                   round(safe_float(latest["open"]),     2),
        "high":                   round(safe_float(latest["high"]),     2),
        "low":                    round(safe_float(latest["low"]),      2),
        "volume":                 int(safe_float(latest["volume"])),
        "prev_close":             round(safe_float(prev["close"]),      2),
        "daily_return_pct":       round(safe_float(latest["daily_return"]), 4),
        "ma_7":                   round(safe_float(latest["ma_7"]),     2),
        "ma_30":                  round(safe_float(latest["ma_30"]),    2),
        "high_52w":               round(df["high_52w"].max(),           2),
        "low_52w":                round(df["low_52w"].min(),            2),
        "volatility_score_30d":   round(safe_float(latest["volatility_30d"]), 4),
        "price_momentum_30d_pct": round(safe_float(latest["price_momentum"]), 4),
    }


# ─────────────────────────────────────────────
# ML PREDICTION  (Linear Regression)
# ─────────────────────────────────────────────

@app.get("/stocks/{symbol}/predict", tags=["ML Prediction"])
def predict_stock_price(
    symbol: str,
    days: int = Query(default=7, ge=1, le=30, description="Days ahead to predict (1–30)"),
):
    """
    **Trend-based demo prediction** using Linear Regression.

    - Trains on the last 90 days of closing prices
    - Returns R² score + a plain-English note on what it means
    - Includes a low-confidence warning if R² < 0.5

    > ⚠️ R² measures fit quality on PAST data, not future accuracy.
    > This is a demonstration only — not investment advice.
    """
    try:
        result = predict_prices(symbol.upper(), days_ahead=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# ─────────────────────────────────────────────
# MARKET
# ─────────────────────────────────────────────

@app.get("/market/gainers-losers", tags=["Market"])
def gainers_losers(
    n: int = Query(default=5, ge=1, le=10, description="Number of top/bottom stocks"),
):
    """
    Top-N gainers and losers for the most recent trading day,
    ranked by `daily_return` (%).
    """
    try:
        result = get_top_gainers_losers(n=n)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return result


@app.get("/market/correlation", tags=["Market"])
def stock_correlation(
    symbol1: str = Query(..., description="First symbol  e.g. TCS"),
    symbol2: str = Query(..., description="Second symbol e.g. INFY"),
    days: int    = Query(default=90, ge=30, le=365, description="Lookback days"),
):
    """
    **Pearson correlation** of daily returns between two stocks.

    - `1.0`  → move in perfect sync
    - `-1.0` → move in opposite directions
    - `≈ 0`  → no relationship

    Returns a friendly error if the same symbol is passed twice,
    or if either symbol is not found.
    """
    try:
        result = get_correlation(symbol1.upper(), symbol2.upper(), days=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Calculation error: {exc}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# ─────────────────────────────────────────────
# ADVANCED ML
# ─────────────────────────────────────────────

@app.get("/stocks/{symbol}/predict/lstm", tags=["Advanced ML"])
def predict_lstm_route(
    symbol: str,
    days: int = Query(default=7, ge=1, le=30, description="Days ahead to predict"),
):
    """
    **LSTM Deep Learning** price prediction.

    Architecture: `LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)`

    - Trained on the last 120 days with a 30-day look-back window
    - Reports MAE and RMSE on a held-out 20% test set
    - First call takes ~20–40 s while the model trains

    > ⚠️ Not financial advice.
    """
    try:
        result = predict_lstm(symbol.upper(), days_ahead=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LSTM error: {exc}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/stocks/{symbol}/sentiment", tags=["Advanced ML"])
def sentiment_route(symbol: str):
    """
    **Sentiment analysis** on recent financial news headlines.

    Scores each headline with VADER NLP (compound: -1.0 → +1.0).
    Aggregates into a composite score and maps to a BUY / HOLD / SELL signal.

    Set the `NEWSAPI_KEY` environment variable for live news;
    otherwise curated mock headlines are used.
    """
    try:
        result = analyze_sentiment(symbol.upper())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sentiment error: {exc}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/market/predict-all", tags=["Advanced ML"])
def predict_all_route(
    days: int = Query(default=7, ge=1, le=30, description="Prediction horizon in days"),
):
    """
    Run price prediction for **all stocks** and return a ranked leaderboard.

    Ranks by predicted % change (highest growth first).
    Each entry includes a BUY / HOLD / SELL signal and R² confidence label.

    > ⚠️ Uses trend extrapolation only. Not investment advice.
    """
    try:
        result = predict_all_stocks(days_ahead=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Multi-predict error: {exc}")

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result
