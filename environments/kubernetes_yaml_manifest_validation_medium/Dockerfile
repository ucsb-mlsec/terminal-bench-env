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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages kubernetes

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /workspace/k8s-manifests

COPY k8s-manifests/deployment1.yaml /workspace/k8s-manifests/deployment1.yaml
COPY k8s-manifests/deployment2.yaml /workspace/k8s-manifests/deployment2.yaml
COPY k8s-manifests/deployment3.yaml /workspace/k8s-manifests/deployment3.yaml
COPY k8s-manifests/service.yaml /workspace/k8s-manifests/service.yaml
COPY k8s-manifests/deployment-memory-low.yaml /workspace/k8s-manifests/deployment-memory-low.yaml
COPY k8s-manifests/deployment-memory-high.yaml /workspace/k8s-manifests/deployment-memory-high.yaml
COPY k8s-manifests/invalid-syntax.yaml /workspace/k8s-manifests/invalid-syntax.yaml
COPY k8s-manifests/wrong-apiversion.yaml /workspace/k8s-manifests/wrong-apiversion.yaml
COPY k8s-manifests/configmap.yaml /workspace/k8s-manifests/configmap.yaml
COPY k8s-manifests/deployment-no-selector.yaml /workspace/k8s-manifests/deployment-no-selector.yaml
COPY k8s-manifests/statefulset.yaml /workspace/k8s-manifests/statefulset.yaml

WORKDIR /workspace