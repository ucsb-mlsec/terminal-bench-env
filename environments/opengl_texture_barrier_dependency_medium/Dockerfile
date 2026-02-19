FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/shaders /solution

COPY data/GLSL_REFERENCE.txt /workspace/GLSL_REFERENCE.txt
COPY data/shaders/normal_compute.glsl /workspace/shaders/normal_compute.glsl
COPY data/shaders/blur_filter.glsl /workspace/shaders/blur_filter.glsl
COPY data/shaders/edge_detect.glsl /workspace/shaders/edge_detect.glsl
COPY data/shaders/copy_shader.glsl /workspace/shaders/copy_shader.glsl
COPY data/shaders/particle_update.glsl /workspace/shaders/particle_update.glsl
COPY data/shaders/color_grade.glsl /workspace/shaders/color_grade.glsl

WORKDIR /workspace