#!/usr/bin/env python3

from flask import Flask, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry
import time

app = Flask(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define various metrics
request_counter = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'], registry=registry)
active_connections = Gauge('active_connections', 'Number of active connections', registry=registry)
response_time = Histogram('response_time_seconds', 'Response time in seconds', registry=registry)
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage', registry=registry)
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes', registry=registry)

@app.route('/metrics')
def metrics():
    # PERFORMANCE BUG: Inefficient nested loop generating unnecessary metric labels
    # This simulates expensive computation that blocks the response
    label_data = []
    for i in range(1000):
        for j in range(1000):
            # Wasteful computation that serves no purpose
            label_data.append(f"label_{i}_{j}")
            if len(label_data) > 100000:
                label_data = label_data[-50000:]  # Keep it from growing infinitely
    
    # Update some sample metrics
    request_counter.labels(method='GET', endpoint='/metrics').inc()
    active_connections.set(42)
    cpu_usage.set(65.5)
    memory_usage.set(1024 * 1024 * 512)  # 512 MB
    
    # Generate and return Prometheus metrics
    metrics_output = generate_latest(registry)
    return Response(metrics_output, mimetype='text/plain')

@app.route('/health')
def health():
    return 'OK', 200

if __name__ == '__main__':
    print("Starting metrics server on port 8000...")
    print("Metrics available at http://localhost:8000/metrics")
    app.run(host='0.0.0.0', port=8000, debug=False)