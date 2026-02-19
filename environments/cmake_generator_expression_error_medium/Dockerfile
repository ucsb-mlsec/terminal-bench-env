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
    ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/signal_processor/src

COPY signal_processor/CMakeLists.txt /workspace/signal_processor/CMakeLists.txt
COPY signal_processor/src/CMakeLists.txt /workspace/signal_processor/src/CMakeLists.txt
COPY signal_processor/src/signal_lib.cpp /workspace/signal_processor/src/signal_lib.cpp
COPY signal_processor/src/signal_lib.h /workspace/signal_processor/src/signal_lib.h
COPY signal_processor/src/main.cpp /workspace/signal_processor/src/main.cpp

WORKDIR /workspace