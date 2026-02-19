FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cluster

COPY config/worker_0.conf /workspace/cluster/worker_0.conf
COPY config/worker_1.conf /workspace/cluster/worker_1.conf
COPY config/worker_2.conf /workspace/cluster/worker_2.conf
COPY config/worker_3.conf /workspace/cluster/worker_3.conf

WORKDIR /workspace