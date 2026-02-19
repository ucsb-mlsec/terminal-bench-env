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
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/calculator/.git/rebase-merge /workspace/calculator/.git/refs/heads

COPY calculator/dot-git/config /workspace/calculator/.git/config
COPY calculator/dot-git/HEAD /workspace/calculator/.git/HEAD
COPY calculator/dot-git/rebase-merge/head-name /workspace/calculator/.git/rebase-merge/head-name
COPY calculator/dot-git/rebase-merge/onto /workspace/calculator/.git/rebase-merge/onto
COPY calculator/calculator.py /workspace/calculator/calculator.py
COPY calculator/dot-git/refs/heads/main /workspace/calculator/.git/refs/heads/main
COPY calculator/dot-git/refs/heads/feature-advanced-ops /workspace/calculator/.git/refs/heads/feature-advanced-ops

WORKDIR /workspace