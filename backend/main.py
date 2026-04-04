from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
import pandas as pd
import os

from utils import (
    get_all_symbols,
    get_stock_df,
    predict_prices,
    get_top_gainers_losers,
    get_correlation,
)
from ml_models import predict_lstm, analyze_sentiment, predict_all_stocks

# APP SETUP

APP_VERSION = "1.1.0"

BASE_DIR = os.path.dirname(__file__)
FRONTEND_FILE = os.path.join(BASE_DIR, "index.html")
DB_FILE = os.path.join(BASE_DIR, "database.db")

app = FastAPI(
    title="Stock Data Intelligence Dashboard",
    description="""
A mini financial data platform built with FastAPI, yfinance, and scikit-learn / TensorFlow.

Educational platform only. Not financial advice.
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

# GENERAL

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "ok" if os.path.exists(DB_FILE) else "degraded",
        "version": APP_VERSION,
        "database_exists": os.path.exists(DB_FILE),
        "frontend_exists": os.path.exists(FRONTEND_FILE),
        "db_path": DB_FILE,
        "frontend_path": FRONTEND_FILE,
    }


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    if os.path.exists(FRONTEND_FILE):
        return FileResponse(FRONTEND_FILE)

    return JSONResponse(
        status_code=404,
        content={"message": "Frontend not found. Ensure backend/index.html exists."},
    )


# STOCKS

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
        "symbol": symbol.upper(),
        "days_returned": len(df),
        "data": df.to_dict(orient="records"),
    }


@app.get("/stocks/{symbol}/summary", tags=["Stocks"])
def get_stock_summary(symbol: str):
    """
    Latest key metrics for a stock.
    """
    try:
        df = get_stock_df(symbol.upper(), days=365)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for '{symbol}'.")

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    def safe_float(val, default=0.0):
        try:
            v = float(val)
            return v if pd.notna(v) else default
        except (TypeError, ValueError):
            return default

    return {
        "symbol": symbol.upper(),
        "last_updated": str(latest["date"].date()),
        "current_price": round(safe_float(latest["close"]), 2),
        "open": round(safe_float(latest["open"]), 2),
        "high": round(safe_float(latest["high"]), 2),
        "low": round(safe_float(latest["low"]), 2),
        "volume": int(safe_float(latest["volume"])),
        "prev_close": round(safe_float(prev["close"]), 2),
        "daily_return_pct": round(safe_float(latest["daily_return"]), 4),
        "ma_7": round(safe_float(latest["ma_7"]), 2),
        "ma_30": round(safe_float(latest["ma_30"]), 2),
        "high_52w": round(df["high_52w"].max(), 2),
        "low_52w": round(df["low_52w"].min(), 2),
        "volatility_score_30d": round(safe_float(latest["volatility_30d"]), 4),
        "price_momentum_30d_pct": round(safe_float(latest["price_momentum"]), 4),
    }


# ML PREDICTION

@app.get("/stocks/{symbol}/predict", tags=["ML Prediction"])
def predict_stock_price(
    symbol: str,
    days: int = Query(default=7, ge=1, le=30, description="Days ahead to predict (1–30)"),
):
    try:
        result = predict_prices(symbol.upper(), days_ahead=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# MARKET

@app.get("/market/gainers-losers", tags=["Market"])
def gainers_losers(
    n: int = Query(default=5, ge=1, le=10, description="Number of top/bottom stocks"),
):
    try:
        result = get_top_gainers_losers(n=n)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return result


@app.get("/market/correlation", tags=["Market"])
def stock_correlation(
    symbol1: str = Query(..., description="First symbol e.g. TCS"),
    symbol2: str = Query(..., description="Second symbol e.g. INFY"),
    days: int = Query(default=90, ge=30, le=365, description="Lookback days"),
):
    try:
        result = get_correlation(symbol1.upper(), symbol2.upper(), days=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Calculation error: {exc}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# ADVANCED ML

@app.get("/stocks/{symbol}/predict/lstm", tags=["Advanced ML"])
def predict_lstm_route(
    symbol: str,
    days: int = Query(default=7, ge=1, le=30, description="Days ahead to predict"),
):
    try:
        result = predict_lstm(symbol.upper(), days_ahead=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LSTM error: {exc}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/stocks/{symbol}/sentiment", tags=["Advanced ML"])
def sentiment_route(symbol: str):
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
    try:
        result = predict_all_stocks(days_ahead=days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Multi-predict error: {exc}")

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result
