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

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /workspace/training_resources

COPY config/training_resource_1.json /workspace/training_resources/training_resource_1.json
COPY config/training_resource_2.yaml /workspace/training_resources/training_resource_2.yaml
COPY config/resources_metadata.json /workspace/training_resources/resources_metadata.json
COPY config/vqsr_config.yaml /workspace/training_resources/vqsr_config.yaml
COPY config/legacy_resources.json /workspace/training_resources/legacy_resources.json

WORKDIR /workspace