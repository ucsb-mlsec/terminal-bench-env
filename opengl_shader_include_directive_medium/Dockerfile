FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/shader_project/shaders /workspace/shader_project/common

COPY shader_project/shaders/vertex_shader.glsl /workspace/shader_project/shaders/vertex_shader.glsl
COPY shader_project/shaders/fragment_shader.glsl /workspace/shader_project/shaders/fragment_shader.glsl
COPY shader_project/shaders/geometry_shader.glsl /workspace/shader_project/shaders/geometry_shader.glsl
COPY shader_project/common/transforms.glsl /workspace/shader_project/common/transforms.glsl
COPY shader_project/common/constants.glsl /workspace/shader_project/common/constants.glsl
COPY shader_project/README.md /workspace/shader_project/README.md
COPY shader_project/STRUCTURE.txt /workspace/shader_project/STRUCTURE.txt

WORKDIR /workspace