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
