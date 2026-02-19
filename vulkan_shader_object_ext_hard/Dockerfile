FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/shaders /workspace/data

COPY data/validation_report.json /workspace/validation_report.json
COPY shaders/vertex_basic.vert /workspace/shaders/vertex_basic.vert
COPY shaders/fragment_pbr.frag /workspace/shaders/fragment_pbr.frag
COPY shaders/compute_particles.comp /workspace/shaders/compute_particles.comp
COPY shaders/vertex_skinned.vert /workspace/shaders/vertex_skinned.vert
COPY shaders/fragment_simple.frag /workspace/shaders/fragment_simple.frag
COPY shaders/vertex_terrain.vert /workspace/shaders/vertex_terrain.vert
COPY shaders/fragment_postprocess.frag /workspace/shaders/fragment_postprocess.frag
COPY shaders/compute_culling.comp /workspace/shaders/compute_culling.comp
COPY shaders/vertex_billboard.vert /workspace/shaders/vertex_billboard.vert

WORKDIR /workspace