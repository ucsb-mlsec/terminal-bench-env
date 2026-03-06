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
    apt-get install -y python3-yaml && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /workspace/shader_configs

COPY config/mountain_terrain.yaml /workspace/shader_configs/mountain_terrain.yaml
COPY config/desert_terrain.yaml /workspace/shader_configs/desert_terrain.yaml
COPY config/ocean_terrain.yaml /workspace/shader_configs/ocean_terrain.yaml
COPY config/forest_terrain.yaml /workspace/shader_configs/forest_terrain.yaml
COPY config/tundra_terrain.yaml /workspace/shader_configs/tundra_terrain.yaml
COPY config/canyon_terrain.yaml /workspace/shader_configs/canyon_terrain.yaml
COPY config/plains_terrain.yaml /workspace/shader_configs/plains_terrain.yaml
COPY config/volcano_terrain.yaml /workspace/shader_configs/volcano_terrain.yaml
COPY config/glacier_terrain.yaml /workspace/shader_configs/glacier_terrain.yaml
COPY config/swamp_terrain.yaml /workspace/shader_configs/swamp_terrain.yaml
COPY config/hills_terrain.yaml /workspace/shader_configs/hills_terrain.yaml
COPY config/valley_terrain.yaml /workspace/shader_configs/valley_terrain.yaml

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
