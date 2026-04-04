FROM python:3.11-slim

WORKDIR /app

# Copy requirements from repo root
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Create data directory
RUN mkdir -p data

CMD ["sh", "-c", "python data.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
