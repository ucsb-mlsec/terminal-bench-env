FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /etc/gpu /var/log/cuda /tmp
RUN mkdir -p /workspace/output/cuda_peer_to_peer_access_medium/files/config
RUN mkdir -p /workspace/output/cuda_peer_to_peer_access_medium/files/data

COPY config/p2p_config.json /etc/gpu/p2p_config.json
COPY data/p2p_status.log /var/log/cuda/p2p_status.log
COPY config/p2p_config.json /workspace/output/cuda_peer_to_peer_access_medium/files/config/p2p_config.json
COPY data/p2p_status.log /workspace/output/cuda_peer_to_peer_access_medium/files/data/p2p_status.log

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
