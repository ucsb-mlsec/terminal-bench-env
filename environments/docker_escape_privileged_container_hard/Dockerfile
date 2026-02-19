FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/container_config /solution

COPY container_config/Dockerfile /workspace/container_config/Dockerfile
COPY container_config/docker-compose.yml /workspace/container_config/docker-compose.yml
COPY container_config/runtime_config.json /workspace/container_config/runtime_config.json
COPY container_config/deployment_notes.txt /workspace/container_config/deployment_notes.txt

WORKDIR /workspace