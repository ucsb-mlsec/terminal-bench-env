FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install headless OpenGL rendering dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    mesa-common-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libosmesa6-dev \
    libglfw3-dev \
    libglew-dev \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential mesa-common-dev libglu1-mesa-dev libglew-dev libglfw3-dev libgl1-mesa-dev libosmesa6-dev freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /workspace/renderer/output /workspace/renderer/reference

# Copy generated files
COPY renderer/render_test.c /workspace/renderer/render_test.c
COPY renderer/Makefile /workspace/renderer/Makefile
COPY renderer/reference/test.ppm /workspace/renderer/reference/test.ppm

WORKDIR /workspace/renderer