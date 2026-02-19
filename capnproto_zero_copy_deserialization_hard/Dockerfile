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
    capnproto \
    libcapnp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y capnproto libcapnp-dev g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/data

COPY message.capnp /workspace/message.capnp
COPY benchmark.cpp /workspace/benchmark.cpp
COPY data/test_data.txt /workspace/test_data.txt
COPY Makefile /workspace/Makefile

WORKDIR /workspace