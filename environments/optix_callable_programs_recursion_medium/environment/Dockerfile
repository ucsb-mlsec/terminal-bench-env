FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/optix_config/test_results/basic_scene && \
    mkdir -p /workspace/optix_config/test_results/mirror_room && \
    mkdir -p /workspace/optix_config/test_results/hall_of_mirrors

COPY config/optix_config/pipeline.yaml /workspace/optix_config/pipeline.yaml
COPY config/optix_config/test_results/basic_scene/metadata.json /workspace/optix_config/test_results/basic_scene/metadata.json
COPY config/optix_config/test_results/mirror_room/metadata.json /workspace/optix_config/test_results/mirror_room/metadata.json
COPY config/optix_config/test_results/hall_of_mirrors/metadata.json /workspace/optix_config/test_results/hall_of_mirrors/metadata.json
COPY config/optix_config/README.md /workspace/optix_config/README.md

WORKDIR /workspace/optix_config

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
