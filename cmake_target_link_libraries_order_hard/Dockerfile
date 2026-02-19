FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    g++ && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake make g++ && \
    rm -rf /var/lib/apt/lists/*

# Create directory structure
RUN mkdir -p /workspace/plugin_system/core && \
    mkdir -p /workspace/plugin_system/plugin && \
    mkdir -p /workspace/plugin_system/app

# Copy all project files
COPY CMakeLists.txt /workspace/plugin_system/CMakeLists.txt
COPY core/CMakeLists.txt /workspace/plugin_system/core/CMakeLists.txt
COPY core/core.hpp /workspace/plugin_system/core/core.hpp
COPY core/core.cpp /workspace/plugin_system/core/core.cpp
COPY plugin/CMakeLists.txt /workspace/plugin_system/plugin/CMakeLists.txt
COPY plugin/plugin.hpp /workspace/plugin_system/plugin/plugin.hpp
COPY plugin/plugin.cpp /workspace/plugin_system/plugin/plugin.cpp
COPY app/CMakeLists.txt /workspace/plugin_system/app/CMakeLists.txt
COPY app/main.cpp /workspace/plugin_system/app/main.cpp

WORKDIR /workspace/plugin_system