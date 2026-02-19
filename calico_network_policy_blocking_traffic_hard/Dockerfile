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
    jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /k8s/policies /solution

COPY k8s/policies/production-default-deny.yaml /k8s/policies/production-default-deny.yaml
COPY k8s/policies/production-frontend-policy.yaml /k8s/policies/production-frontend-policy.yaml
COPY k8s/policies/production-api-policy.yaml /k8s/policies/production-api-policy.yaml
COPY k8s/policies/data-tier-default-deny.yaml /k8s/policies/data-tier-default-deny.yaml
COPY k8s/policies/data-tier-postgres-policy.yaml /k8s/policies/data-tier-postgres-policy.yaml
COPY k8s/policies/monitoring-prometheus-policy.yaml /k8s/policies/monitoring-prometheus-policy.yaml
COPY k8s/policies/production-deny-cross-namespace.yaml /k8s/policies/production-deny-cross-namespace.yaml
COPY k8s/policies/monitoring-scrape-deny.yaml /k8s/policies/monitoring-scrape-deny.yaml
COPY k8s/policies/data-tier-strict-isolation.yaml /k8s/policies/data-tier-strict-isolation.yaml

WORKDIR /workspace