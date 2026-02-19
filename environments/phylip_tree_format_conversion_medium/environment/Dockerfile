FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages biopython

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/trees /workspace/converted

COPY data/trees/tree1.nwk /workspace/trees/tree1.nwk
COPY data/trees/tree2.nwk /workspace/trees/tree2.nwk
COPY data/trees/phylo1.nex /workspace/trees/phylo1.nex
COPY data/trees/phylo2.nex /workspace/trees/phylo2.nex

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
