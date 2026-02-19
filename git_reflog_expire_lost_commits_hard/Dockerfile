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

RUN mkdir -p /workspace/repo/.git/objects/1a && \
    mkdir -p /workspace/repo/.git/objects/2b && \
    mkdir -p /workspace/repo/.git/objects/3c && \
    mkdir -p /workspace/repo/.git/objects/4d && \
    mkdir -p /workspace/repo/.git/objects/5e && \
    mkdir -p /workspace/repo/.git/objects/6f && \
    mkdir -p /workspace/repo/.git/refs/heads

COPY repo/dot-git/config /workspace/repo/.git/config
COPY repo/dot-git/HEAD /workspace/repo/.git/HEAD
COPY repo/dot-git/refs/heads/main /workspace/repo/.git/refs/heads/main
COPY repo/dot-git/objects/1a/2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b /workspace/repo/.git/objects/1a/2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b
COPY repo/dot-git/objects/2b/3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c /workspace/repo/.git/objects/2b/3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c
COPY repo/dot-git/objects/3c/4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d /workspace/repo/.git/objects/3c/4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d
COPY repo/dot-git/objects/4d/5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e /workspace/repo/.git/objects/4d/5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e
COPY repo/dot-git/objects/5e/6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f /workspace/repo/.git/objects/5e/6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f
COPY repo/dot-git/objects/6f/7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a /workspace/repo/.git/objects/6f/7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a
COPY repo/README.md /workspace/repo/README.md

WORKDIR /workspace