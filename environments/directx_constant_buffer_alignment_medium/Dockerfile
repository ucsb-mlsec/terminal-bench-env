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

COPY shaders/BasicLighting.hlsl /workspace/shaders/BasicLighting.hlsl
COPY shaders/MaterialProperties.hlsl /workspace/shaders/MaterialProperties.hlsl
COPY shaders/PostProcess.hlsl /workspace/shaders/PostProcess.hlsl
COPY shaders/TerrainRenderer.hlsl /workspace/shaders/TerrainRenderer.hlsl
COPY shaders/ParticleSystem.hlsl /workspace/shaders/ParticleSystem.hlsl

WORKDIR /workspace