FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/config_system /tmp

COPY config_system/gpu_config.json /workspace/config_system/gpu_config.json
COPY config_system/shader_vertex.glsl /workspace/config_system/shader_vertex.glsl
COPY config_system/shader_fragment.glsl /workspace/config_system/shader_fragment.glsl
COPY config_system/extension_map.txt /workspace/config_system/extension_map.txt

WORKDIR /workspace