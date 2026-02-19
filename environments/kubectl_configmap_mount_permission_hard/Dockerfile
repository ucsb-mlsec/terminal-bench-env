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
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyyaml>=5.1

RUN mkdir -p /kubernetes/deployments /kubernetes/configmaps /solution/deployments

COPY kubernetes/deployments/frontend-deployment.yaml /kubernetes/deployments/frontend-deployment.yaml
COPY kubernetes/deployments/backend-deployment.yaml /kubernetes/deployments/backend-deployment.yaml
COPY kubernetes/deployments/worker-deployment.yaml /kubernetes/deployments/worker-deployment.yaml
COPY kubernetes/configmaps/common-settings.yaml /kubernetes/configmaps/common-settings.yaml
COPY kubernetes/configmaps/frontend-config.yaml /kubernetes/configmaps/frontend-config.yaml
COPY kubernetes/configmaps/backend-config.yaml /kubernetes/configmaps/backend-config.yaml
COPY kubernetes/configmaps/backend-db-config.yaml /kubernetes/configmaps/backend-db-config.yaml
COPY kubernetes/configmaps/backend-secrets.yaml /kubernetes/configmaps/backend-secrets.yaml
COPY kubernetes/configmaps/worker-config.yaml /kubernetes/configmaps/worker-config.yaml
COPY kubernetes/configmaps/worker-tasks.yaml /kubernetes/configmaps/worker-tasks.yaml

WORKDIR /workspace