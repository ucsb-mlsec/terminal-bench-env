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
    build-essential \
    gcc \
    g++ \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake clang build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/legacy-app/build

COPY legacy-app/main.cpp /workspace/legacy-app/main.cpp
COPY legacy-app/utils.cpp /workspace/legacy-app/utils.cpp
COPY legacy-app/utils.h /workspace/legacy-app/utils.h
COPY legacy-app/CMakeLists.txt /workspace/legacy-app/CMakeLists.txt

WORKDIR /workspace