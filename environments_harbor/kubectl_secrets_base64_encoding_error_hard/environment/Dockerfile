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
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /workspace/k8s-secrets /workspace/app-configs

COPY k8s-secrets/database-secret.yaml /workspace/k8s-secrets/database-secret.yaml
COPY k8s-secrets/api-keys-secret.yaml /workspace/k8s-secrets/api-keys-secret.yaml
COPY k8s-secrets/tls-cert-secret.yaml /workspace/k8s-secrets/tls-cert-secret.yaml
COPY k8s-secrets/config-json-secret.yaml /workspace/k8s-secrets/config-json-secret.yaml
COPY k8s-secrets/redis-secret.yaml /workspace/k8s-secrets/redis-secret.yaml
COPY k8s-secrets/oauth-secret.yaml /workspace/k8s-secrets/oauth-secret.yaml
COPY k8s-secrets/jwt-secret.yaml /workspace/k8s-secrets/jwt-secret.yaml
COPY k8s-secrets/monitoring-secret.yaml /workspace/k8s-secrets/monitoring-secret.yaml
COPY expected-values.txt /workspace/expected-values.txt
COPY app-configs/database-app-config.yaml /workspace/app-configs/database-app-config.yaml
COPY app-configs/api-service-config.yaml /workspace/app-configs/api-service-config.yaml
COPY app-configs/web-server-config.yaml /workspace/app-configs/web-server-config.yaml
COPY app-configs/config-loader-app.yaml /workspace/app-configs/config-loader-app.yaml

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
