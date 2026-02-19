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
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/project

COPY math_functions.cpp /home/project/math_functions.cpp
COPY math_functions.h /home/project/math_functions.h
COPY test_add.cpp /home/project/test_add.cpp
COPY test_multiply.cpp /home/project/test_multiply.cpp
COPY test_edge_cases.cpp /home/project/test_edge_cases.cpp
COPY CMakeLists.txt /home/project/CMakeLists.txt

WORKDIR /home/project