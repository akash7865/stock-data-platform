"""
ml_models.py — Advanced ML Models for Stock Prediction
────────────────────────────────────────────────────────
Three models:

1. LSTM (Deep Learning)
   - Uses last 60 days of close prices as sequences
   - Predicts next N days via a 2-layer LSTM + Dense network
   - Built with TensorFlow/Keras

2. Sentiment Analysis
   - Fetches recent financial news headlines via NewsAPI (or mock)
   - Scores each headline with VADER (rule-based NLP)
   - Aggregates into a composite sentiment score per stock
   - Maps score → BUY / HOLD / SELL signal

3. Multi-Stock Prediction
   - Runs Linear Regression for all stocks in one call
   - Returns ranked predictions: which stocks are predicted
     to grow the most over the next N days
   - Useful for portfolio-level decision making
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils import get_stock_df, get_all_symbols

# ═══════════════════════════════════════════════════════════
# 1. LSTM DEEP LEARNING PREDICTION
# ═══════════════════════════════════════════════════════════

def predict_lstm(symbol: str, days_ahead: int = 7) -> dict:
    """
    Predict stock closing prices using a 2-layer LSTM neural network.

    Architecture:
      Input  →  LSTM(64)  →  Dropout(0.2)
             →  LSTM(32)  →  Dropout(0.2)
             →  Dense(1)  →  Output

    Training:
      - Uses last 120 days of close prices
      - Sequence length (look_back) = 30 days
      - Normalized with MinMaxScaler
      - Trained for 30 epochs with early stopping

    Returns historical (last 30 days) + predicted prices.
    """
    # ── Import guards ──────────────────────────────────────
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        return {
            "error": (
                "TensorFlow not installed. Run: pip install tensorflow "
                "(or use the Linear Regression endpoint instead)"
            )
        }

    # ── Load data ──────────────────────────────────────────
    df = get_stock_df(symbol, days=200)
    if len(df) < 60:
        return {"error": f"Need at least 60 days of data for LSTM. Got {len(df)} for {symbol}."}

    close_prices = df["close"].values.reshape(-1, 1)

    # ── Scale to [0, 1] ────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    # ── Build sequences ────────────────────────────────────
    # Each X[i] = 30 days of prices → y[i] = next day price
    LOOK_BACK = 30
    X, y = [], []
    for i in range(LOOK_BACK, len(scaled)):
        X.append(scaled[i - LOOK_BACK:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, LOOK_BACK, 1)  # (samples, timesteps, features)
    y = np.array(y)

    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ── Build LSTM model ───────────────────────────────────
    tf.random.set_seed(42)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0,
    )

    # ── Evaluate on test set ───────────────────────────────
    test_pred = model.predict(X_test, verbose=0)
    test_pred_inv = scaler.inverse_transform(test_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = float(np.mean(np.abs(test_pred_inv - y_test_inv)))
    rmse = float(np.sqrt(np.mean((test_pred_inv - y_test_inv) ** 2)))

    # ── Predict future days ────────────────────────────────
    # Start with the last LOOK_BACK days and iterate forward
    last_sequence = scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
    predicted_scaled = []

    current_seq = last_sequence.copy()
    for _ in range(days_ahead):
        next_val = model.predict(current_seq, verbose=0)[0, 0]
        predicted_scaled.append(next_val)
        # Slide window: drop first, append new prediction
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 0] = next_val

    predicted_prices = scaler.inverse_transform(
        np.array(predicted_scaled).reshape(-1, 1)
    ).flatten().round(2).tolist()

    # ── Future dates (weekdays only) ───────────────────────
    last_date = df["date"].iloc[-1]
    future_dates = []
    current = last_date
    while len(future_dates) < days_ahead:
        current += timedelta(days=1)
        if current.weekday() < 5:
            future_dates.append(current.strftime("%Y-%m-%d"))

    return {
        "symbol": symbol,
        "model": "LSTM (2-layer Deep Learning)",
        "architecture": "LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(1)",
        "look_back_days": LOOK_BACK,
        "evaluation": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "note": "Evaluated on 20% held-out test data",
        },
        "note": "Deep learning prediction based on price sequences. Not financial advice.",
        "historical": {
            "dates": df["date"].dt.strftime("%Y-%m-%d").tolist()[-30:],
            "prices": df["close"].round(2).tolist()[-30:],
        },
        "prediction": {
            "dates": future_dates,
            "prices": predicted_prices,
        },
    }


# ═══════════════════════════════════════════════════════════
# 2. SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════

# Mock headlines when no NewsAPI key is set.
# In production replace with: requests.get(NEWSAPI_URL, params={...})
MOCK_HEADLINES = {
    "RELIANCE": [
        "Reliance Industries posts record quarterly profit, beats estimates",
        "Reliance Jio 5G expansion accelerates across India",
        "Mukesh Ambani unveils new green energy investment of ₹75,000 crore",
        "Reliance Retail surpasses targets; analysts upgrade stock",
        "Reliance faces regulatory scrutiny in telecom pricing",
    ],
    "TCS": [
        "TCS wins $1.5B multi-year deal with European bank",
        "TCS Q3 revenue growth beats Street estimates",
        "TCS plans to hire 40,000 freshers in FY25",
        "TCS launches AI-powered platform for enterprise clients",
        "Attrition at TCS drops to 12.5%, lowest in 6 quarters",
    ],
    "INFY": [
        "Infosys raises FY25 revenue guidance after strong Q2",
        "Infosys wins deal with US healthcare firm worth $500M",
        "Infosys Topaz AI platform sees rising enterprise adoption",
        "Infosys cuts jobs amid restructuring concerns",
        "Infosys CFO warns of macro headwinds in Europe",
    ],
    "HDFCBANK": [
        "HDFC Bank net profit grows 18% YoY in Q3 FY25",
        "HDFC Bank increases deposit rates, attracting retail savers",
        "HDFC Bank launches co-branded credit card with Marriott",
        "HDFC Bank NPA ratio marginally rises; analysts cautious",
        "RBI lifts business restrictions on HDFC Bank credit cards",
    ],
    "ICICIBANK": [
        "ICICI Bank reports 15% profit jump in Q3",
        "ICICI Bank digital lending grows 40% YoY",
        "ICICI Bank expands rural banking network to 3000 villages",
        "ICICI Bank faces higher provisioning amid unsecured loan stress",
        "ICICI Bank wins Best Digital Bank award in Asia",
    ],
    "WIPRO": [
        "Wipro bags $700M deal with UK utilities company",
        "Wipro Q3 revenue misses estimates amid weak demand",
        "Wipro CEO outlines AI-first transformation strategy",
        "Wipro attrition stabilizes; talent crunch easing",
        "Wipro faces margin pressure from wage hikes",
    ],
    "SBIN": [
        "SBI posts highest ever quarterly profit of ₹16,884 crore",
        "SBI home loan book grows 14% in Q3",
        "SBI launches digital rupee pilot for retail customers",
        "SBI gross NPA rises slightly; management says under control",
        "SBI Chairman bullish on credit growth for FY25",
    ],
    "BAJFINANCE": [
        "Bajaj Finance AUM crosses ₹3.5 lakh crore milestone",
        "Bajaj Finance Q3 PAT rises 21% YoY, beats estimates",
        "Bajaj Finance expands into insurance and wealth management",
        "Bajaj Finance raises concern over credit card slippages",
        "Analysts cut target price on Bajaj Finance on NPA worries",
    ],
    "ADANIENT": [
        "Adani Enterprises wins renewable energy tender worth ₹12,000 crore",
        "Adani Group announces airport expansion in 5 cities",
        "Adani stock rebounds after Hindenburg-related sell-off subsides",
        "Adani Enterprises Q3 net profit falls 6% YoY",
        "SEBI probe into Adani creates uncertainty: analysts",
    ],
    "TATAMOTORS": [
        "Tata Motors EV sales hit 50,000 units in a single month",
        "Tata Motors JLR division posts record profit on strong demand",
        "Tata Motors launches Harrier EV; pre-bookings cross 10,000",
        "Tata Motors commercial vehicle volumes dip 8% in Q3",
        "Tata Motors stock up 40% in 2024 on EV momentum",
    ],
}

# Fallback headlines for unknown symbols
DEFAULT_HEADLINES = [
    "Stock market sees broad-based rally amid positive global cues",
    "Investors cautious ahead of RBI monetary policy meeting",
    "FII inflows positive for third consecutive week",
    "Nifty 50 touches new all-time high on strong earnings season",
    "Midcap stocks underperform as profit booking sets in",
]


def analyze_sentiment(symbol: str) -> dict:
    """
    Perform sentiment analysis on financial news headlines for a stock.

    Method:
      1. Fetch headlines (mock data or NewsAPI if key is set)
      2. Score each headline using VADER SentimentIntensityAnalyzer
         VADER is rule-based NLP tuned for financial/social text
      3. Compute composite score (mean of compound scores)
      4. Map to trading signal: BUY / HOLD / SELL

    VADER compound score:
      +1.0 = most positive   |   -1.0 = most negative   |   0 = neutral
    Signal thresholds:
      compound >= 0.15  → BUY
      compound <= -0.15 → SELL
      else              → HOLD
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return {
            "error": (
                "vaderSentiment not installed. Run: pip install vaderSentiment"
            )
        }

    analyzer = SentimentIntensityAnalyzer()

    # Get headlines — use NewsAPI if key exists, else mock data
    newsapi_key = os.environ.get("NEWSAPI_KEY", "")
    headlines = []

    if newsapi_key:
        try:
            import requests
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": symbol + " stock India",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "apiKey": newsapi_key,
            }
            resp = requests.get(url, params=params, timeout=5)
            articles = resp.json().get("articles", [])
            headlines = [a["title"] for a in articles if a.get("title")]
        except Exception:
            headlines = []  # fall through to mock

    if not headlines:
        headlines = MOCK_HEADLINES.get(symbol.upper(), DEFAULT_HEADLINES)
        data_source = "mock headlines (set NEWSAPI_KEY env var for real news)"
    else:
        data_source = "NewsAPI (live)"

    # ── Score each headline ────────────────────────────────
    scored = []
    for h in headlines:
        scores = analyzer.polarity_scores(h)
        scored.append({
            "headline": h,
            "positive": round(scores["pos"], 3),
            "negative": round(scores["neg"], 3),
            "neutral":  round(scores["neu"], 3),
            "compound": round(scores["compound"], 4),
        })

    # ── Aggregate ──────────────────────────────────────────
    compound_scores = [s["compound"] for s in scored]
    mean_compound = round(float(np.mean(compound_scores)), 4)
    positive_count = sum(1 for c in compound_scores if c >= 0.05)
    negative_count = sum(1 for c in compound_scores if c <= -0.05)
    neutral_count  = len(compound_scores) - positive_count - negative_count

    # ── Signal mapping ─────────────────────────────────────
    if mean_compound >= 0.15:
        signal = "BUY"
        signal_color = "green"
        signal_reason = "Predominantly positive news sentiment detected."
    elif mean_compound <= -0.15:
        signal = "SELL"
        signal_color = "red"
        signal_reason = "Predominantly negative news sentiment detected."
    else:
        signal = "HOLD"
        signal_color = "yellow"
        signal_reason = "Mixed or neutral sentiment. Insufficient directional signal."

    return {
        "symbol": symbol,
        "signal": signal,
        "signal_color": signal_color,
        "signal_reason": signal_reason,
        "composite_score": mean_compound,
        "score_range": "[-1.0 = very negative → +1.0 = very positive]",
        "headline_summary": {
            "total": len(scored),
            "positive": positive_count,
            "neutral":  neutral_count,
            "negative": negative_count,
        },
        "headlines": scored,
        "data_source": data_source,
        "note": "Sentiment is based on news headlines only. Not financial advice.",
    }


# ═══════════════════════════════════════════════════════════
# 3. MULTI-STOCK PREDICTION (Ranking)
# ═══════════════════════════════════════════════════════════

def predict_all_stocks(days_ahead: int = 7) -> dict:
    """
    Run price prediction for every stock in the database and rank them
    by predicted percentage growth over the next N days.

    Method:
      - Uses Linear Regression per stock (fast, no GPU needed)
      - Predicts last price vs price N days ahead
      - Ranks by predicted % change (descending)

    This gives a portfolio-level view:
      "Which stocks are predicted to grow the most?"

    Note: We use Linear Regression here (not LSTM) so this endpoint
    responds quickly for all stocks simultaneously.
    """
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return {"error": "scikit-learn not installed. Run: pip install scikit-learn"}

    symbols = get_all_symbols()
    results = []

    for symbol in symbols:
        try:
            df = get_stock_df(symbol, days=90)
            if len(df) < 20:
                continue

            df = df.reset_index(drop=True)
            df["day_index"] = np.arange(len(df))

            X = df[["day_index"]].values
            y = df["close"].values

            model = LinearRegression()
            model.fit(X, y)

            last_idx = df["day_index"].iloc[-1]
            current_price = float(df["close"].iloc[-1])

            # Predict N trading days ahead
            future_idx = np.array([[last_idx + days_ahead]])
            predicted_price = float(model.predict(future_idx)[0])
            predicted_change_pct = round(
                (predicted_price - current_price) / current_price * 100, 4
            )

            # R² score for model confidence
            r2 = round(model.score(X, y), 4)

            # Confidence label based on R²
            if r2 >= 0.85:
                confidence = "High"
            elif r2 >= 0.60:
                confidence = "Medium"
            else:
                confidence = "Low"

            results.append({
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "predicted_price": round(predicted_price, 2),
                "predicted_change_pct": predicted_change_pct,
                "r2_score": r2,
                "confidence": confidence,
                "signal": "BUY" if predicted_change_pct > 1.0
                          else "SELL" if predicted_change_pct < -1.0
                          else "HOLD",
            })

        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})

    # Rank by predicted change (highest growth first)
    valid = [r for r in results if "predicted_change_pct" in r]
    valid.sort(key=lambda x: x["predicted_change_pct"], reverse=True)

    errors = [r for r in results if "error" in r]

    # Generate future target date
    last_date = datetime.today()
    count = 0
    while count < days_ahead:
        last_date += timedelta(days=1)
        if last_date.weekday() < 5:
            count += 1
    target_date = last_date.strftime("%Y-%m-%d")

    return {
        "prediction_horizon_days": days_ahead,
        "target_date": target_date,
        "model": "Linear Regression (per stock)",
        "ranked_predictions": valid,
        "top_pick": valid[0] if valid else None,
        "worst_pick": valid[-1] if valid else None,
        "errors": errors if errors else None,
        "note": (
            "Rankings based on price trend extrapolation only. "
            "High R² = model fits historical trend well. "
            "This is NOT investment advice."
        ),
    }
