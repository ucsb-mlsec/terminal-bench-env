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

RUN mkdir -p /workspace/loglib/src /workspace/loglib/include

COPY loglib/src/logger.cpp /workspace/loglib/src/logger.cpp
COPY loglib/include/logger.h /workspace/loglib/include/logger.h
COPY loglib/src/main.cpp /workspace/loglib/src/main.cpp
COPY loglib/CMakeLists.txt /workspace/loglib/CMakeLists.txt
COPY loglib/README.md /workspace/loglib/README.md

WORKDIR /workspace