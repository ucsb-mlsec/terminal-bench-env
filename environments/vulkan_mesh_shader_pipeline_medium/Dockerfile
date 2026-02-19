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

COPY data/shaders/vertex_main.spv /workspace/shaders/vertex_main.spv
COPY data/shaders/fragment_main.spv /workspace/shaders/fragment_main.spv
COPY data/shaders/compute_pass1.spv /workspace/shaders/compute_pass1.spv
COPY data/shaders/geometry_pass.spv /workspace/shaders/geometry_pass.spv
COPY data/shaders/corrupted_vertex.spv /workspace/shaders/corrupted_vertex.spv
COPY data/shaders/test_fragment.spv /workspace/shaders/test_fragment.spv
COPY data/shaders/invalid_compute.spv /workspace/shaders/invalid_compute.spv
COPY data/shaders/tessellation_control.spv /workspace/shaders/tessellation_control.spv
COPY data/shaders/old_backup.spv /workspace/shaders/old_backup.spv
COPY data/shaders/raytracing_gen.spv /workspace/shaders/raytracing_gen.spv

WORKDIR /workspace