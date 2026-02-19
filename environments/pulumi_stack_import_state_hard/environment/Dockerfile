FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/inventory \
    /workspace/pulumi-project/.pulumi \
    /workspace/solution

COPY data/inventory/cloud_resources.json /workspace/inventory/cloud_resources.json
COPY data/inventory/type_mappings.json /workspace/inventory/type_mappings.json
COPY data/pulumi-project/Pulumi.yaml /workspace/pulumi-project/Pulumi.yaml
COPY data/pulumi-project/Pulumi.prod.yaml /workspace/pulumi-project/Pulumi.prod.yaml

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
