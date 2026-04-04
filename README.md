# 📈 StockSense — Stock Data Intelligence Dashboard

> 🚀 A full-stack financial data platform built using FastAPI, Machine Learning, and interactive visualization.
> 🎯 Developed for the **Jarnox Internship Assignment**

---

## 🌐 Live Demo (Coming Soon)

> Deploying soon on Render / Railway 🚀

---

## 🖼️ Project Preview

* 📊 Interactive stock charts
* 📈 Real-time-like NSE data insights
* 🤖 ML-based predictions
* 📉 Market gainers & losers

---

## 🧱 Project Architecture

```
Frontend (HTML + Chart.js)
        ↓
FastAPI Backend (REST APIs)
        ↓
Data Pipeline (yfinance + Pandas)
        ↓
SQLite Database
        ↓
ML Models (Linear Regression + LSTM + Sentiment)
```

---

## ⚙️ Tech Stack

| Layer     | Technology                      |
| --------- | ------------------------------- |
| Backend   | FastAPI, Uvicorn                |
| Data      | Pandas, NumPy, yfinance         |
| Database  | SQLite                          |
| ML Models | scikit-learn, TensorFlow (LSTM) |
| NLP       | VADER Sentiment                 |
| Frontend  | HTML, CSS, JavaScript, Chart.js |
| DevOps    | Docker, Docker Compose          |
| Docs      | Swagger UI                      |

---

## 🚀 Features

### 📊 Data Processing

* Cleaned and structured real NSE stock data
* Handled missing values and invalid formats
* Converted date formats properly

### 📈 Financial Metrics

* Daily Return
* Moving Averages (7-day, 30-day)
* 52-week High / Low
* Volatility Score
* Price Momentum

### 🔗 Advanced Analytics

* Stock correlation analysis
* Top gainers & losers
* Multi-stock ranking

### 🤖 Machine Learning

* Linear Regression (trend prediction)
* LSTM Deep Learning model
* Sentiment Analysis (BUY / HOLD / SELL signals)

### 🌐 REST API

* Fully structured FastAPI backend
* 10+ endpoints
* Swagger documentation (`/docs`)

### 💻 Dashboard

* Interactive charts (Chart.js)
* Company watchlist
* Time filters (1M, 3M, 6M, 1Y)
* Live-style UI with insights

---

## 🚀 Local Setup

### 1. Clone Repository

```bash
git clone https://github.com/akash7865/stock-data-platform.git
cd stock-data-platform
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run Data Pipeline

```bash
python backend/data.py
```

---

### 4. Start Backend

```bash
cd backend
uvicorn main:app --reload
```

---

### 5. Open Dashboard

👉 http://localhost:8000/dashboard

---

## 🐳 Docker Setup

```bash
docker-compose up --build
```

---

## 📡 Key API Endpoints

| Endpoint                     | Description        |
| ---------------------------- | ------------------ |
| `/stocks`                    | List all stocks    |
| `/stocks/{symbol}`           | Historical data    |
| `/stocks/{symbol}/summary`   | Key metrics        |
| `/stocks/{symbol}/predict`   | ML prediction      |
| `/market/gainers-losers`     | Market movers      |
| `/market/correlation`        | Stock correlation  |
| `/stocks/{symbol}/sentiment` | Sentiment analysis |

---

## 💡 Creativity & Enhancements

This project goes beyond basic requirements by including:

* ✅ Deep Learning (LSTM prediction)
* ✅ Sentiment-based trading signals
* ✅ Market ranking system
* ✅ Clean UI dashboard
* ✅ Dockerized backend
* ✅ Structured architecture

---

## ⚠️ Disclaimer

> This is an educational project.
> All predictions are for demonstration only and not financial advice.

---

## 👨‍💻 Author

**Akash Dhara**
GitHub: https://github.com/akash7865

---

## 🎯 Why This Project Stands Out

* Real-world data handling
* Full-stack implementation
* ML + Analytics integration
* Clean architecture
* Production-ready API design

---

⭐ If you found this project useful, consider starring the repo!
