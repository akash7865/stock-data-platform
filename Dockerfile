# ─────────────────────────────────────────────
# Dockerfile — Stock Data Intelligence Dashboard
# ─────────────────────────────────────────────

# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (layer caching — faster rebuilds)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create data directory
RUN python backend/data.py

# Run data pipeline first, then start API server
# The CMD will: (1) fetch & build DB, (2) start FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Expose port 8000
EXPOSE 8000
