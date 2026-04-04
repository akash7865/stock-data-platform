FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy BOTH backend and frontend
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Move into backend
WORKDIR /app/backend

# Run data pipeline + start server
CMD ["sh", "-c", "python data.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
