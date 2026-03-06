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
    cmake \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev mesa-utils libglfw3-dev libglew-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/render_app/shaders \
    /workspace/render_app/textures \
    /workspace/render_app/config \
    /workspace/render_app/logs \
    /workspace/render_app/output

COPY render_app/main.cpp /workspace/render_app/main.cpp
COPY render_app/stb_image_write.h /workspace/render_app/stb_image_write.h
COPY render_app/CMakeLists.txt /workspace/render_app/CMakeLists.txt
COPY config/framebuffer.conf /workspace/render_app/config/framebuffer.conf
COPY render_app/shaders/vertex.glsl /workspace/render_app/shaders/vertex.glsl
COPY render_app/shaders/fragment.glsl /workspace/render_app/shaders/fragment.glsl
COPY render_app/textures/test_texture.png /workspace/render_app/textures/test_texture.png

WORKDIR /workspace/render_app

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
