FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN pip3 install --no-cache-dir --break-system-packages numpy pandas

RUN mkdir -p /data/experiments /solution /output

COPY data/experiment_001.npy /data/experiments/experiment_001.npy
COPY data/experiment_002.npy /data/experiments/experiment_002.npy
COPY data/experiment_003.npy /data/experiments/experiment_003.npy

WORKDIR /workspace