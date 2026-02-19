FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/k8s-diagnostics/pod-logs

COPY data/k8s-diagnostics/pods.json /workspace/k8s-diagnostics/pods.json
COPY data/k8s-diagnostics/events.json /workspace/k8s-diagnostics/events.json
COPY data/k8s-diagnostics/pod-logs/production-api-server-7d8f9.log /workspace/k8s-diagnostics/pod-logs/production-api-server-7d8f9.log
COPY data/k8s-diagnostics/pod-logs/staging-worker-queue-3k2m1.log /workspace/k8s-diagnostics/pod-logs/staging-worker-queue-3k2m1.log
COPY data/k8s-diagnostics/pod-logs/default-cache-redis-8x4n2.log /workspace/k8s-diagnostics/pod-logs/default-cache-redis-8x4n2.log
COPY data/k8s-diagnostics/pod-logs/production-frontend-9m3x7.log /workspace/k8s-diagnostics/pod-logs/production-frontend-9m3x7.log
COPY data/k8s-diagnostics/pod-logs/staging-auth-service-5k8p2.log /workspace/k8s-diagnostics/pod-logs/staging-auth-service-5k8p2.log

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
