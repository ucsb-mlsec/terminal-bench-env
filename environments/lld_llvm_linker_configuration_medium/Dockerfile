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
    clang \
    lld \
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential lld make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/cpp_project

COPY cpp_project/main.cpp /workspace/cpp_project/main.cpp
COPY cpp_project/math_ops.cpp /workspace/cpp_project/math_ops.cpp
COPY cpp_project/string_utils.cpp /workspace/cpp_project/string_utils.cpp
COPY cpp_project/Makefile /workspace/cpp_project/Makefile

WORKDIR /workspace