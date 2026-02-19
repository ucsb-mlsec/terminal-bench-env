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
    apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/matrix_calc \
    /workspace/build \
    /opt/eigen-3.4.0/share/eigen3/cmake \
    /opt/eigen-3.4.0/include/eigen3/Eigen

COPY matrix_calc/CMakeLists.txt /workspace/matrix_calc/CMakeLists.txt
COPY matrix_calc/main.cpp /workspace/matrix_calc/main.cpp
COPY opt/eigen-3.4.0/share/eigen3/cmake/Eigen3Config.cmake /opt/eigen-3.4.0/share/eigen3/cmake/Eigen3Config.cmake
COPY opt/eigen-3.4.0/include/eigen3/Eigen/Dense /opt/eigen-3.4.0/include/eigen3/Eigen/Dense
COPY opt/eigen-3.4.0/include/eigen3/Eigen/Core /opt/eigen-3.4.0/include/eigen3/Eigen/Core

WORKDIR /workspace