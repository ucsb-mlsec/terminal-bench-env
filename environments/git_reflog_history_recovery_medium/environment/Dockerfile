FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y git && \
    pip3 install --no-cache-dir --break-system-packages flask && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/payment-service/.git/logs/refs/heads && \
    mkdir -p /workspace

COPY workspace/payment-service/dot-git/config /workspace/payment-service/.git/config
COPY workspace/payment-service/dot-git/HEAD /workspace/payment-service/.git/HEAD
COPY workspace/payment-service/dot-git/logs/HEAD /workspace/payment-service/.git/logs/HEAD
COPY workspace/payment-service/dot-git/logs/refs/heads/main /workspace/payment-service/.git/logs/refs/heads/main
COPY workspace/payment-service/dot-git/logs/refs/heads/feature/payment-gateway /workspace/payment-service/.git/logs/refs/heads/feature/payment-gateway
COPY workspace/payment-service/README.md /workspace/payment-service/README.md
COPY workspace/payment-service/main.py /workspace/payment-service/main.py

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
