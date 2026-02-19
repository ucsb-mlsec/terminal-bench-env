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
    build-essential \
    g++ \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    libglm-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace /solution /data

COPY renderer.cpp /workspace/renderer.cpp
COPY data/buffer_layout.json /workspace/data/buffer_layout.json
COPY mesh_generator.cpp /workspace/mesh_generator.cpp

WORKDIR /workspace