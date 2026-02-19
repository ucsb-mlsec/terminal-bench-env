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
    golang-go \
    gcc \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc g++ libc6-dev make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/calculator

COPY calculator/main.go /workspace/calculator/main.go
COPY calculator/calculator.c /workspace/calculator/calculator.c
COPY calculator/calculator.h /workspace/calculator/calculator.h
COPY calculator/go.mod /workspace/calculator/go.mod

WORKDIR /workspace/calculator