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
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/graphics_engine/data \
    /workspace/graphics_engine/config \
    /tmp

COPY data/debug_spec.txt /workspace/graphics_engine/data/debug_spec.txt
COPY config/debug_config.ini /workspace/graphics_engine/config/debug_config.ini
COPY debug_callback.cpp /workspace/graphics_engine/debug_callback.cpp
COPY callback_manager.h /workspace/graphics_engine/callback_manager.h

WORKDIR /workspace