FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install necessary build tools and linker
RUN apt-get update && \
    apt-get install -y \
    clang \
    lld \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y clang lld make libc++-dev libc++abi-dev && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directory structure
RUN mkdir -p /workspace/src \
    /workspace/lib/math \
    /workspace/lib/utils \
    /workspace/bin \
    /workspace/build

# Copy source files
COPY src/main.cpp /workspace/src/main.cpp
COPY lib/math/operations.cpp /workspace/lib/math/operations.cpp
COPY lib/math/operations.h /workspace/lib/math/operations.h
COPY lib/utils/helpers.cpp /workspace/lib/utils/helpers.cpp
COPY lib/utils/helpers.h /workspace/lib/utils/helpers.h

# Copy build configuration
COPY Makefile /workspace/Makefile

WORKDIR /workspace