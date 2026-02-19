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
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /k8s-manifests /solution

COPY k8s-manifests/auth-service.yaml /k8s-manifests/auth-service.yaml
COPY k8s-manifests/api-gateway.yaml /k8s-manifests/api-gateway.yaml
COPY k8s-manifests/user-service.yaml /k8s-manifests/user-service.yaml
COPY k8s-manifests/payment-service.yaml /k8s-manifests/payment-service.yaml
COPY k8s-manifests/ingress.yaml /k8s-manifests/ingress.yaml

WORKDIR /k8s-manifests