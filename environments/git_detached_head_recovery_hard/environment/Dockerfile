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

RUN mkdir -p /workspace/project_repo/.git/refs/heads \
    /workspace/project_repo/.git/objects/info \
    /workspace/project_repo/.git/logs/refs/heads

COPY project_repo/dot-git/HEAD /workspace/project_repo/.git/HEAD
COPY project_repo/dot-git/refs/heads/main /workspace/project_repo/.git/refs/heads/main
COPY project_repo/dot-git/objects/info/.gitkeep /workspace/project_repo/.git/objects/info/.gitkeep
COPY project_repo/dot-git/logs/HEAD /workspace/project_repo/.git/logs/HEAD
COPY project_repo/dot-git/logs/refs/heads/main /workspace/project_repo/.git/logs/refs/heads/main
COPY project_repo/dot-git/config /workspace/project_repo/.git/config
COPY project_repo/README.md /workspace/project_repo/README.md

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
