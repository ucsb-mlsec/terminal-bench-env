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
    g++ \
    ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y ninja-build && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/project/src /workspace/project/build

COPY project/src/generator.cpp /workspace/project/src/generator.cpp
COPY project/src/main.cpp /workspace/project/src/main.cpp
COPY project/build/build.ninja /workspace/project/build/build.ninja

WORKDIR /workspace