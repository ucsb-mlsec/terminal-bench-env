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
    apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pandas

RUN mkdir -p /workspace/logs

COPY data/logs/app_access_2024_01.jsonl /workspace/logs/app_access_2024_01.jsonl
COPY data/logs/app_access_2024_02.jsonl /workspace/logs/app_access_2024_02.jsonl
COPY data/logs/app_access_2024_03.jsonl /workspace/logs/app_access_2024_03.jsonl

WORKDIR /workspace