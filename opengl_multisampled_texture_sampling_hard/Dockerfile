FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/shaders

COPY data/shaders/forward_render.shader /workspace/shaders/forward_render.shader
COPY data/shaders/deferred_gbuffer.shader /workspace/shaders/deferred_gbuffer.shader
COPY data/shaders/shadow_map.shader /workspace/shaders/shadow_map.shader
COPY data/shaders/post_process_bloom.shader /workspace/shaders/post_process_bloom.shader
COPY data/shaders/ssao_ambient.shader /workspace/shaders/ssao_ambient.shader
COPY data/shaders/particle_renderer.shader /workspace/shaders/particle_renderer.shader
COPY data/shaders/volumetric_fog.shader /workspace/shaders/volumetric_fog.shader
COPY data/shaders/terrain_splat.shader /workspace/shaders/terrain_splat.shader
COPY data/shaders/water_reflection.shader /workspace/shaders/water_reflection.shader
COPY data/shaders/ui_overlay.shader /workspace/shaders/ui_overlay.shader
COPY data/shaders/skybox_cubemap.shader /workspace/shaders/skybox_cubemap.shader
COPY data/shaders/decal_projector.shader /workspace/shaders/decal_projector.shader

WORKDIR /workspace