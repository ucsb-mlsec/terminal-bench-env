FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages fastapi==0.104.1 uvicorn==0.24.0 pydantic==2.5.0 opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi opentelemetry-exporter-otlp

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /app /solution

COPY app/requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt && \
    pip3 install --no-cache-dir --break-system-packages \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-exporter-otlp

WORKDIR /app

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
