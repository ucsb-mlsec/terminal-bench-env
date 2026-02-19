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
    apt-get install -y build-essential g++ make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/pch_project/src /workspace/pch_project/include

COPY pch_project/src/main.cpp /workspace/pch_project/src/main.cpp
COPY pch_project/src/operations.cpp /workspace/pch_project/src/operations.cpp
COPY pch_project/src/utilities.cpp /workspace/pch_project/src/utilities.cpp
COPY pch_project/include/common.h /workspace/pch_project/include/common.h
COPY pch_project/Makefile /workspace/pch_project/Makefile

WORKDIR /workspace/pch_project