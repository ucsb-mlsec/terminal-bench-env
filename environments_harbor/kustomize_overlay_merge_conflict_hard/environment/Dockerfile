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
    apt-get install -y curl && \
    curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash && \
    mv kustomize /usr/local/bin/ && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyyaml>=5.1

RUN mkdir -p /workspace/base \
    /workspace/overlays/dev \
    /workspace/overlays/staging \
    /workspace/overlays/prod \
    /solution

COPY base/deployment.yaml /workspace/base/deployment.yaml
COPY base/service.yaml /workspace/base/service.yaml
COPY base/configmap.yaml /workspace/base/configmap.yaml
COPY base/kustomization.yaml /workspace/base/kustomization.yaml

COPY overlays/dev/kustomization.yaml /workspace/overlays/dev/kustomization.yaml
COPY overlays/dev/deployment-patch.yaml /workspace/overlays/dev/deployment-patch.yaml

COPY overlays/staging/kustomization.yaml /workspace/overlays/staging/kustomization.yaml
COPY overlays/staging/deployment-patch.yaml /workspace/overlays/staging/deployment-patch.yaml
COPY overlays/staging/memory-patch.json /workspace/overlays/staging/memory-patch.json
COPY overlays/staging/cpu-patch.json /workspace/overlays/staging/cpu-patch.json
COPY overlays/staging/configmap-patch.yaml /workspace/overlays/staging/configmap-patch.yaml

COPY overlays/prod/kustomization.yaml /workspace/overlays/prod/kustomization.yaml
COPY overlays/prod/deployment-patch.yaml /workspace/overlays/prod/deployment-patch.yaml
COPY overlays/prod/service-patch.yaml /workspace/overlays/prod/service-patch.yaml

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
