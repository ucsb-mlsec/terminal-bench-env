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
    mpich \
    libmpich-dev && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake mpich g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/matrix_compute/src /opt/matrix_compute/build

COPY opt/matrix_compute/CMakeLists.txt /opt/matrix_compute/CMakeLists.txt
COPY opt/matrix_compute/src/matrix_compute.cpp /opt/matrix_compute/src/matrix_compute.cpp

WORKDIR /opt/matrix_compute