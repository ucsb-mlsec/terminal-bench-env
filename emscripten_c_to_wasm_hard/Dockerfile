FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Emscripten dependencies and tools
RUN apt-get update && \
    apt-get install -y \
    git \
    cmake \
    python3 \
    python3-pip \
    nodejs \
    npm \
    build-essential \
    wget \
    xz-utils && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y emscripten && \
    rm -rf /var/lib/apt/lists/*

# Install Emscripten SDK
RUN cd /opt && \
    git clone https://github.com/emscripten-core/emsdk.git && \
    cd emsdk && \
    ./emsdk install latest && \
    ./emsdk activate latest

# Add Emscripten to PATH
ENV PATH="/opt/emsdk:/opt/emsdk/upstream/emscripten:${PATH}"
ENV EMSDK="/opt/emsdk"
ENV EM_CONFIG="/opt/emsdk/.emscripten"

# Create workspace directory structure
RUN mkdir -p /workspace/math_lib

# Copy project files to container
COPY math_lib/mathutils.c /workspace/math_lib/mathutils.c
COPY math_lib/test.html /workspace/math_lib/test.html

WORKDIR /workspace