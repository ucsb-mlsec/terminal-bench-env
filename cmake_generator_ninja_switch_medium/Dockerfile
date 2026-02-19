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
    apt-get install -y cmake ninja-build g++ && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/cpp_project/build

COPY cpp_project/CMakeLists.txt /workspace/cpp_project/CMakeLists.txt
COPY cpp_project/main.cpp /workspace/cpp_project/main.cpp
COPY cpp_project/build/CMakeCache.txt /workspace/cpp_project/build/CMakeCache.txt
COPY cpp_project/build/Makefile /workspace/cpp_project/build/Makefile

WORKDIR /workspace