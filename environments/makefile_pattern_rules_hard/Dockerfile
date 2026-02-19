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
RUN apt-get update && apt-get install -y build-essential g++ make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/src /workspace/include

COPY src/main.cpp /workspace/src/main.cpp
COPY src/utils.cpp /workspace/src/utils.cpp
COPY src/parser.cpp /workspace/src/parser.cpp
COPY include/utils.hpp /workspace/include/utils.hpp
COPY include/parser.hpp /workspace/include/parser.hpp
COPY Makefile /workspace/Makefile

WORKDIR /workspace