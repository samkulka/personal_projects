# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install build tools needed for numpy / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY optimizer.py fastapi_app.py ml_workloads.py /app/

EXPOSE 8080
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8080"]
