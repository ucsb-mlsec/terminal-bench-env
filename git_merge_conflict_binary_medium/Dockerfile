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

RUN mkdir -p /workspace/repo/.git /workspace/repo/assets /workspace/repo/config

COPY repo/dot-git/MERGE_HEAD /workspace/repo/.git/MERGE_HEAD
COPY repo/dot-git/MERGE_MSG /workspace/repo/.git/MERGE_MSG
COPY repo/dot-git/index /workspace/repo/.git/index
COPY repo/dot-git/config /workspace/repo/.git/config
COPY repo/dot-git/HEAD /workspace/repo/.git/HEAD
COPY repo/assets/banner.png /workspace/repo/assets/banner.png
COPY repo/assets/icon.png /workspace/repo/assets/icon.png
COPY repo/config/database.conf /workspace/repo/config/database.conf
COPY repo/config/settings.conf /workspace/repo/config/settings.conf
COPY repo/README.md /workspace/repo/README.md

WORKDIR /workspace