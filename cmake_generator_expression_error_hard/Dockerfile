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
    make \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user/mathlib_project/src \
    /home/user/mathlib_project/test \
    /home/user/mathlib_project/build

COPY home/user/mathlib_project/CMakeLists.txt /home/user/mathlib_project/CMakeLists.txt
COPY home/user/mathlib_project/src/mathlib.h /home/user/mathlib_project/src/mathlib.h
COPY home/user/mathlib_project/src/mathlib.cpp /home/user/mathlib_project/src/mathlib.cpp
COPY home/user/mathlib_project/test/test_main.cpp /home/user/mathlib_project/test/test_main.cpp

WORKDIR /home/user/mathlib_project