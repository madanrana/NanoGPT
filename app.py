# app.py
from fastapi import FastAPI
from prometheus_client import start_http_server, Summary
import time
import threading

app = FastAPI()

# Dummy health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Prometheus metrics
REQUEST_TIME = Summary('nanogpt_training_seconds', 'Time spent per training step')

@app.get("/train_step")
def train_step():
    # Example step, replace with actual training logic
    time.sleep(0.5)
    REQUEST_TIME.observe(0.5)
    return {"status": "step_completed"}

# Start Prometheus metrics server on a separate port
def start_prometheus_server():
    start_http_server(8001)  # Prometheus will scrape this port

threading.Thread(target=start_prometheus_server, daemon=True).start()