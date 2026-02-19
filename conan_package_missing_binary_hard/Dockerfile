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
    cmake \
    g++ \
    gcc \
    python3-pip \
    libpng-dev \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake gcc g++ make libpng-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages conan

RUN mkdir -p /workspace/imageproc/src \
    /workspace/imageproc/profiles

COPY workspace/imageproc/conanfile.txt /workspace/imageproc/conanfile.txt
COPY workspace/imageproc/CMakeLists.txt /workspace/imageproc/CMakeLists.txt
COPY workspace/imageproc/src/processor.h /workspace/imageproc/src/processor.h
COPY workspace/imageproc/src/processor.cpp /workspace/imageproc/src/processor.cpp
COPY workspace/imageproc/profiles/ubuntu2004.profile /workspace/imageproc/profiles/ubuntu2004.profile
COPY workspace/imageproc/profiles/debian11.profile /workspace/imageproc/profiles/debian11.profile
COPY workspace/imageproc/profiles/centos8.profile /workspace/imageproc/profiles/centos8.profile
COPY workspace/test_image.bmp /workspace/test_image.bmp

WORKDIR /workspace