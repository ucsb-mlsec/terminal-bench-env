FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data /config /tmp

COPY data/pbs_queue_state.json /data/pbs_queue_state.json
COPY config/pbs_server_config.txt /config/pbs_server_config.txt
COPY config/fairshare_config.txt /config/fairshare_config.txt
COPY config/queue_priorities.conf /config/queue_priorities.conf

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
