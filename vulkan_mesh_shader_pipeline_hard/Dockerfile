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
    spirv-tools \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y glslang-tools && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/shader_project

COPY shader_project/shader.vert /workspace/shader_project/shader.vert
COPY shader_project/shader.frag /workspace/shader_project/shader.frag
COPY shader_project/interface_spec.txt /workspace/shader_project/interface_spec.txt

WORKDIR /workspace