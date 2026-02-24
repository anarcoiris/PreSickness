FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt
RUN pip install --no-cache-dir --default-timeout=100 fastapi uvicorn pydantic

# Copy source code
COPY . .

# Default command
CMD ["uvicorn", "tinyllm_app:app", "--host", "0.0.0.0", "--port", "8003"]
