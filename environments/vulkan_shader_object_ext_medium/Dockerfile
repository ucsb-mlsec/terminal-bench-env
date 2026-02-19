FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/shader_configs

COPY data/shader_configs/config1.json /workspace/shader_configs/config1.json
COPY data/shader_configs/config2.json /workspace/shader_configs/config2.json
COPY data/shader_configs/config3.json /workspace/shader_configs/config3.json
COPY data/shader_configs/config4.json /workspace/shader_configs/config4.json
COPY data/shader_configs/config5.json /workspace/shader_configs/config5.json
COPY data/shader_configs/config6.json /workspace/shader_configs/config6.json
COPY data/shader_configs/config7.json /workspace/shader_configs/config7.json
COPY data/shader_configs/config8.json /workspace/shader_configs/config8.json

WORKDIR /workspace