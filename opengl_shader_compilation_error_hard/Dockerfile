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
    apt-get install -y --no-install-recommends \
    glslang-tools \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y glslang-tools && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/render_engine/shaders /workspace/render_engine/docs

COPY render_engine/shaders/transform.vert /workspace/render_engine/shaders/transform.vert
COPY render_engine/shaders/lighting.frag /workspace/render_engine/shaders/lighting.frag
COPY render_engine/docs/shader_spec.txt /workspace/render_engine/docs/shader_spec.txt

WORKDIR /workspace