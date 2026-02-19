FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/shader_validation/shaders

COPY data/shaders/terrain.hlsl /workspace/shader_validation/shaders/terrain.hlsl
COPY data/shaders/lighting.hlsl /workspace/shader_validation/shaders/lighting.hlsl
COPY data/shaders/postprocess.hlsl /workspace/shader_validation/shaders/postprocess.hlsl
COPY data/shaders/particle.hlsl /workspace/shader_validation/shaders/particle.hlsl
COPY data/shaders/skybox.hlsl /workspace/shader_validation/shaders/skybox.hlsl
COPY data/shaders/ui.hlsl /workspace/shader_validation/shaders/ui.hlsl
COPY data/shaders/water.hlsl /workspace/shader_validation/shaders/water.hlsl
COPY data/shaders/shadow.hlsl /workspace/shader_validation/shaders/shadow.hlsl
COPY data/binding_rules.txt /workspace/shader_validation/binding_rules.txt
COPY data/project_config.json /workspace/shader_validation/project_config.json

WORKDIR /workspace/shader_validation