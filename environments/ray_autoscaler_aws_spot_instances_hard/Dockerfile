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
    apt-get install -y python3-pip python3-yaml && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages ray[default]

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /workspace /workspace/solution /workspace/data

COPY ray_cluster.yaml /workspace/ray_cluster.yaml
COPY data/ray_config_reference.txt /workspace/ray_config_reference.txt
COPY data/available_instances.json /workspace/available_instances.json

WORKDIR /workspace