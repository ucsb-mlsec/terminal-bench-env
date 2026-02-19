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

RUN mkdir -p /workspace/project/.git/logs \
    /workspace/project/.git/refs/heads \
    /workspace/project/.git/objects/pack \
    /workspace/project/.git/objects/info \
    /workspace/project/src

COPY project/dot-git/HEAD /workspace/project/.git/HEAD
COPY project/dot-git/logs/HEAD /workspace/project/.git/logs/HEAD
COPY project/dot-git/config /workspace/project/.git/config
COPY project/dot-git/refs/heads/main /workspace/project/.git/refs/heads/main
COPY project/dot-git/objects/pack/.gitkeep /workspace/project/.git/objects/pack/.gitkeep
COPY project/dot-git/objects/info/.gitkeep /workspace/project/.git/objects/info/.gitkeep
COPY project/README.md /workspace/project/README.md
COPY project/src/main.py /workspace/project/src/main.py

WORKDIR /workspace