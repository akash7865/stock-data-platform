FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

WORKDIR /app/backend

CMD ["sh", "-c", "python data.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
