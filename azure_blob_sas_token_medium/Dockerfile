FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages pyyaml

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/config

COPY config/azure_config.json /workspace/config/azure_config.json
COPY config/azure_connection.yaml /workspace/config/azure_connection.yaml
COPY config/storage.env /workspace/config/storage.env
COPY config/blob_endpoints.txt /workspace/config/blob_endpoints.txt
COPY config/deployment_config.json /workspace/config/deployment_config.json
COPY config/legacy_settings.yaml /workspace/config/legacy_settings.yaml

WORKDIR /workspace