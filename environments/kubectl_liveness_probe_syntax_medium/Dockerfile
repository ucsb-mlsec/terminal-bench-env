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
    apt-get install -y curl wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml kubernetes

RUN mkdir -p /workspace/k8s-manifests

COPY k8s-manifests/frontend-deployment.yaml /workspace/k8s-manifests/frontend-deployment.yaml
COPY k8s-manifests/backend-deployment.yaml /workspace/k8s-manifests/backend-deployment.yaml
COPY k8s-manifests/db-proxy-deployment.yaml /workspace/k8s-manifests/db-proxy-deployment.yaml

WORKDIR /workspace