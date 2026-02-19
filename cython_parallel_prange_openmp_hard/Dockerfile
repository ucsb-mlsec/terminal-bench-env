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
    cmake \
    make \
    libomp-dev \
    pybind11-dev \
    python3-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pybind11 numpy

RUN mkdir -p /workspace/numlib/src

COPY src/compute.cpp /workspace/numlib/src/compute.cpp
COPY benchmark.py /workspace/numlib/benchmark.py
COPY build.sh /workspace/numlib/build.sh

RUN chmod +x /workspace/numlib/build.sh

WORKDIR /workspace/numlib