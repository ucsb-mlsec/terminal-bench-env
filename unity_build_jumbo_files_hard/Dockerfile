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
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/calculator/src \
    /workspace/calculator/include

COPY workspace/calculator/src/addition.cpp /workspace/calculator/src/addition.cpp
COPY workspace/calculator/src/subtraction.cpp /workspace/calculator/src/subtraction.cpp
COPY workspace/calculator/src/multiplication.cpp /workspace/calculator/src/multiplication.cpp
COPY workspace/calculator/src/division.cpp /workspace/calculator/src/division.cpp
COPY workspace/calculator/include/addition.h /workspace/calculator/include/addition.h
COPY workspace/calculator/include/subtraction.h /workspace/calculator/include/subtraction.h
COPY workspace/calculator/include/multiplication.h /workspace/calculator/include/multiplication.h
COPY workspace/calculator/include/division.h /workspace/calculator/include/division.h
COPY workspace/calculator/Makefile /workspace/calculator/Makefile
COPY workspace/calculator/test_main.cpp /workspace/calculator/test_main.cpp

WORKDIR /workspace