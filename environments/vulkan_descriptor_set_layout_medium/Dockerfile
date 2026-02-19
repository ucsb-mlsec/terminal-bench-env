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
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y spirv-tools && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace /workspace/shaders /workspace/config

COPY shaders/vertex_shader.glsl /workspace/shaders/vertex_shader.glsl
COPY shaders/fragment_shader.glsl /workspace/shaders/fragment_shader.glsl
COPY config/descriptor_layout.json /workspace/config/descriptor_layout.json
COPY config/pipeline_config.json /workspace/config/pipeline_config.json
COPY validation_errors.log /workspace/validation_errors.log

WORKDIR /workspace