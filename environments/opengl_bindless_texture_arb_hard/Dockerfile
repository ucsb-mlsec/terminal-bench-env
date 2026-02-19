FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/graphics_engine/shaders /workspace/graphics_engine/src

COPY graphics_engine/shaders/material_pbr.frag /workspace/graphics_engine/shaders/material_pbr.frag
COPY graphics_engine/shaders/skybox.frag /workspace/graphics_engine/shaders/skybox.frag
COPY graphics_engine/shaders/terrain.frag /workspace/graphics_engine/shaders/terrain.frag
COPY graphics_engine/shaders/post_process.frag /workspace/graphics_engine/shaders/post_process.frag
COPY graphics_engine/shaders/particle.frag /workspace/graphics_engine/shaders/particle.frag
COPY graphics_engine/src/texture_manager.cpp /workspace/graphics_engine/src/texture_manager.cpp
COPY graphics_engine/src/renderer_init.cpp /workspace/graphics_engine/src/renderer_init.cpp

WORKDIR /workspace