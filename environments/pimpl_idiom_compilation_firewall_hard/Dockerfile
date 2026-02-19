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
    apt-get install -y --no-install-recommends \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/graphics/include /workspace/graphics/src /workspace/graphics/test

COPY graphics/include/Canvas.h /workspace/graphics/include/Canvas.h
COPY graphics/include/Palette.h /workspace/graphics/include/Palette.h
COPY graphics/include/Pipeline.h /workspace/graphics/include/Pipeline.h
COPY graphics/src/Canvas.cpp /workspace/graphics/src/Canvas.cpp
COPY graphics/src/Palette.cpp /workspace/graphics/src/Palette.cpp
COPY graphics/src/Pipeline.cpp /workspace/graphics/src/Pipeline.cpp
COPY graphics/test/main.cpp /workspace/graphics/test/main.cpp
COPY graphics/Makefile /workspace/graphics/Makefile

WORKDIR /workspace