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
    apt-get install -y clang-14 llvm-14 && \
    ln -sf /usr/bin/clang-14 /usr/bin/clang && \
    ln -sf /usr/bin/clang++-14 /usr/bin/clang++ && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y clang libc6-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/jsonlib

COPY jsonlib/json_parser.h /workspace/jsonlib/json_parser.h
COPY jsonlib/json_parser.c /workspace/jsonlib/json_parser.c
COPY jsonlib/Makefile /workspace/jsonlib/Makefile

WORKDIR /workspace