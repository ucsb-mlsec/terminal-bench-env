FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /opt/prometheus/metrics_samples /solution

COPY config/prometheus.yml /opt/prometheus/prometheus.yml
COPY data/cardinality_report.txt /opt/prometheus/cardinality_report.txt
COPY data/metrics_samples/api-gateway.txt /opt/prometheus/metrics_samples/api-gateway.txt
COPY data/metrics_samples/user-service.txt /opt/prometheus/metrics_samples/user-service.txt
COPY data/metrics_samples/payment-processor.txt /opt/prometheus/metrics_samples/payment-processor.txt
COPY data/metrics_samples/session-manager.txt /opt/prometheus/metrics_samples/session-manager.txt
COPY data/metrics_samples/data-pipeline.txt /opt/prometheus/metrics_samples/data-pipeline.txt

WORKDIR /opt/prometheus