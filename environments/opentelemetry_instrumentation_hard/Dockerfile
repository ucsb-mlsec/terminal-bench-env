FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages opentelemetry-api==1.20.0 opentelemetry-sdk==1.20.0 opentelemetry-exporter-otlp-proto-http==1.20.0

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /app/service /app/output /solution

RUN pip3 install --no-cache-dir --break-system-packages \
    opentelemetry-api==1.20.0 \
    opentelemetry-sdk==1.20.0

COPY app/service/processor.py /app/service/processor.py
COPY app/test_runner.py /app/test_runner.py

WORKDIR /app