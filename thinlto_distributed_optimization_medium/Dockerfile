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
    clang \
    lld \
    llvm \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake clang lld build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/cpp_project/src

COPY workspace/cpp_project/CMakeLists.txt /workspace/cpp_project/CMakeLists.txt
COPY workspace/cpp_project/src/main.cpp /workspace/cpp_project/src/main.cpp
COPY workspace/cpp_project/src/factorial.cpp /workspace/cpp_project/src/factorial.cpp
COPY workspace/cpp_project/src/factorial.h /workspace/cpp_project/src/factorial.h
COPY workspace/cpp_project/src/fibonacci.cpp /workspace/cpp_project/src/fibonacci.cpp
COPY workspace/cpp_project/src/fibonacci.h /workspace/cpp_project/src/fibonacci.h
COPY workspace/cpp_project/src/prime.cpp /workspace/cpp_project/src/prime.cpp
COPY workspace/cpp_project/src/prime.h /workspace/cpp_project/src/prime.h

WORKDIR /workspace/cpp_project