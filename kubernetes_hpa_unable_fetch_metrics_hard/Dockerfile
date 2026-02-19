FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyyaml

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cluster-state /solution

COPY data/cluster-state/metrics-server.yaml /workspace/cluster-state/metrics-server.yaml
COPY data/cluster-state/rbac.yaml /workspace/cluster-state/rbac.yaml
COPY data/cluster-state/apiservice.yaml /workspace/cluster-state/apiservice.yaml
COPY data/cluster-state/sample-hpa.yaml /workspace/cluster-state/sample-hpa.yaml

WORKDIR /workspace