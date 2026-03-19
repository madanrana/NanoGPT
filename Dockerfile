# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy the entire project
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn prometheus_client torch numpy

# Expose ports
EXPOSE 8000  # FastAPI / health
EXPOSE 8001  # Prometheus metrics

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]