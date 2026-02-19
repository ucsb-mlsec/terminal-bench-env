FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/configs /workspace/reference

COPY data/vk_ext_descriptor_buffer_spec.txt /workspace/reference/vk_ext_descriptor_buffer_spec.txt
COPY config/graphics_pipeline.json /workspace/configs/graphics_pipeline.json
COPY config/compute_shader.json /workspace/configs/compute_shader.json
COPY config/material_system.json /workspace/configs/material_system.json
COPY config/texture_sampler.json /workspace/configs/texture_sampler.json
COPY config/lighting_pass.json /workspace/configs/lighting_pass.json
COPY config/particle_system.json /workspace/configs/particle_system.json
COPY config/shadow_mapping.json /workspace/configs/shadow_mapping.json
COPY config/post_processing.json /workspace/configs/post_processing.json
COPY config/ui_renderer.json /workspace/configs/ui_renderer.json
COPY config/terrain_rendering.json /workspace/configs/terrain_rendering.json

WORKDIR /workspace