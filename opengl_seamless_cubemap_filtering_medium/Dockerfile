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
    gcc \
    make \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/graphics_project/src

COPY graphics_project/src/main.c /workspace/graphics_project/src/main.c
COPY graphics_project/src/renderer.c /workspace/graphics_project/src/renderer.c
COPY graphics_project/src/texture.c /workspace/graphics_project/src/texture.c
COPY graphics_project/src/shader.c /workspace/graphics_project/src/shader.c
COPY graphics_project/src/camera.c /workspace/graphics_project/src/camera.c
COPY graphics_project/src/utils.c /workspace/graphics_project/src/utils.c

WORKDIR /workspace