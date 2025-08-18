FROM python:3.11-slim

WORKDIR /app

# System deps (optional but helpful)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose FastAPI and MLflow typical ports
EXPOSE 8000 5000 4200 9696

# Default command: run API
CMD ["uvicorn", "code.api:app", "--host", "0.0.0.0", "--port", "8000"]
