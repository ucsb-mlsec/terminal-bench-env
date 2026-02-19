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

RUN mkdir -p /workspace/cluster-config /root/solution

COPY config/namespaces.yaml /workspace/cluster-config/namespaces.yaml
COPY config/deployments.yaml /workspace/cluster-config/deployments.yaml
COPY config/mutatingwebhook.yaml /workspace/cluster-config/mutatingwebhook.yaml
COPY config/configmap.yaml /workspace/cluster-config/configmap.yaml
COPY data/istiod-logs.txt /workspace/cluster-config/istiod-logs.txt

WORKDIR /workspace