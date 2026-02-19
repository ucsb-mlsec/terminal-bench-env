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

RUN mkdir -p /workspace/project/.git \
    /workspace/project/lib/shared/.git/objects/a1 \
    /workspace/project/lib/shared/.git/objects/x9 \
    /workspace/project/lib/shared/.git/refs/heads

COPY workspace/project/dot-git/config /workspace/project/.git/config
COPY workspace/project/dot-git/HEAD /workspace/project/.git/HEAD
COPY workspace/project/dot-git/index /workspace/project/.git/index
COPY workspace/project/.gitmodules /workspace/project/.gitmodules
COPY workspace/project/lib/shared/dot-git/config /workspace/project/lib/shared/.git/config
COPY workspace/project/lib/shared/dot-git/HEAD /workspace/project/lib/shared/.git/HEAD
COPY workspace/project/lib/shared/dot-git/refs/heads/main /workspace/project/lib/shared/.git/refs/heads/main
COPY workspace/project/lib/shared/dot-git/objects/a1/b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0 /workspace/project/lib/shared/.git/objects/a1/b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0
COPY workspace/project/lib/shared/dot-git/objects/x9/y8z7w6v5u4t3s2r1q0p9o8n7m6l5k4j3i2h1g0 /workspace/project/lib/shared/.git/objects/x9/y8z7w6v5u4t3s2r1q0p9o8n7m6l5k4j3i2h1g0
COPY workspace/project/README.md /workspace/project/README.md
COPY workspace/project/lib/shared/shared.py /workspace/project/lib/shared/shared.py

WORKDIR /workspace/project