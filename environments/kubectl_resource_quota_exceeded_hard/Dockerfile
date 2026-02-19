FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/k8s-manifests /tmp

COPY data/k8s-manifests/frontend-namespace.yaml /home/agent/k8s-manifests/frontend-namespace.yaml
COPY data/k8s-manifests/backend-namespace.yaml /home/agent/k8s-manifests/backend-namespace.yaml
COPY data/k8s-manifests/data-namespace.yaml /home/agent/k8s-manifests/data-namespace.yaml
COPY data/k8s-manifests/frontend-limitrange.yaml /home/agent/k8s-manifests/frontend-limitrange.yaml
COPY data/k8s-manifests/backend-limitrange.yaml /home/agent/k8s-manifests/backend-limitrange.yaml
COPY data/k8s-manifests/data-limitrange.yaml /home/agent/k8s-manifests/data-limitrange.yaml
COPY data/k8s-manifests/frontend-pods.yaml /home/agent/k8s-manifests/frontend-pods.yaml
COPY data/k8s-manifests/backend-pods.yaml /home/agent/k8s-manifests/backend-pods.yaml
COPY data/k8s-manifests/data-pods.yaml /home/agent/k8s-manifests/data-pods.yaml
COPY data/k8s-manifests/mixed-units.yaml /home/agent/k8s-manifests/mixed-units.yaml

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

WORKDIR /home/agent