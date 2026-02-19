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

RUN mkdir -p /workspace/project-repo/.git/objects/78 && \
    mkdir -p /workspace/project-repo/.git/objects/ab && \
    mkdir -p /workspace/project-repo/.git/objects/12 && \
    mkdir -p /workspace/project-repo/.git/logs

COPY project-repo/dot-git/config /workspace/project-repo/.git/config
COPY project-repo/dot-git/HEAD /workspace/project-repo/.git/HEAD
COPY project-repo/dot-git/logs/HEAD /workspace/project-repo/.git/logs/HEAD
COPY project-repo/dot-git/objects/78/9012345678901234567890abcdef123456a /workspace/project-repo/.git/objects/78/9012345678901234567890abcdef123456a
COPY project-repo/dot-git/objects/ab/c123def456789012345678901234567890abcd /workspace/project-repo/.git/objects/ab/c123def456789012345678901234567890abcd
COPY project-repo/dot-git/objects/12/3456789012345678901234567890abcdef1234 /workspace/project-repo/.git/objects/12/3456789012345678901234567890abcdef1234

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
