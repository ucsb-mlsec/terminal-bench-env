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
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cmake g++ make pkg-config findutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/calculator && \
    mkdir -p /opt/calcops_custom/include && \
    mkdir -p /opt/calcops_custom/lib/pkgconfig

COPY workspace/calculator/main.cpp /workspace/calculator/main.cpp
COPY workspace/calculator/CMakeLists.txt /workspace/calculator/CMakeLists.txt
COPY opt/calcops_custom/include/calcops.h /opt/calcops_custom/include/calcops.h
COPY opt/calcops_custom/lib/libcalcops.so.1.0 /opt/calcops_custom/lib/libcalcops.so.1.0
COPY opt/calcops_custom/lib/pkgconfig/calcops.pc /opt/calcops_custom/lib/pkgconfig/calcops.pc

RUN chmod 755 /opt/calcops_custom/lib/libcalcops.so.1.0

WORKDIR /workspace