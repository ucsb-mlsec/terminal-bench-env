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
    g++ \
    libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake g++ make libomp-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user/matrix_compute/build

COPY home/user/matrix_compute/CMakeLists.txt /home/user/matrix_compute/CMakeLists.txt
COPY home/user/matrix_compute/main.cpp /home/user/matrix_compute/main.cpp
COPY home/user/matrix_compute/matrix_ops.cpp /home/user/matrix_compute/matrix_ops.cpp
COPY home/user/matrix_compute/matrix_ops.h /home/user/matrix_compute/matrix_ops.h

WORKDIR /home/user/matrix_compute