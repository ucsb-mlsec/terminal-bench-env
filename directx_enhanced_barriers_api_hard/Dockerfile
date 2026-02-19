FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/renderer

COPY renderer/geometry_processor.cpp /workspace/renderer/geometry_processor.cpp
COPY renderer/texture_manager.cpp /workspace/renderer/texture_manager.cpp
COPY renderer/compute_pipeline.cpp /workspace/renderer/compute_pipeline.cpp
COPY renderer/render_pass.h /workspace/renderer/render_pass.h
COPY renderer/resource_uploader.cpp /workspace/renderer/resource_uploader.cpp
COPY renderer/shadow_mapper.cpp /workspace/renderer/shadow_mapper.cpp
COPY renderer/post_process.cpp /workspace/renderer/post_process.cpp

WORKDIR /workspace