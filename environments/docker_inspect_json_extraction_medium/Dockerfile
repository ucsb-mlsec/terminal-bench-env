FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/docker_data

COPY data/docker_data/container_webserver.json /workspace/docker_data/container_webserver.json
COPY data/docker_data/container_database.json /workspace/docker_data/container_database.json
COPY data/docker_data/container_cache.json /workspace/docker_data/container_cache.json
COPY data/docker_data/container_api.json /workspace/docker_data/container_api.json
COPY data/docker_data/container_worker.json /workspace/docker_data/container_worker.json
COPY data/docker_data/container_monitor.json /workspace/docker_data/container_monitor.json

WORKDIR /workspace