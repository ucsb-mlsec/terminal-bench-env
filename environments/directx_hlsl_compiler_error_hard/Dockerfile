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
    apt-get install -y \
    glslang-tools \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y glslang-tools && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/shaders

COPY shaders/basic.vert /workspace/shaders/basic.vert
COPY shaders/basic.frag /workspace/shaders/basic.frag
COPY shaders/terrain.vert /workspace/shaders/terrain.vert
COPY shaders/terrain.frag /workspace/shaders/terrain.frag
COPY shaders/particle.geom /workspace/shaders/particle.geom
COPY shaders/skybox.vert /workspace/shaders/skybox.vert
COPY shaders/skybox.frag /workspace/shaders/skybox.frag

WORKDIR /workspace