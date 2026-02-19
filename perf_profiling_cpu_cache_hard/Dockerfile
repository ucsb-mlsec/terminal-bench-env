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
    gcc \
    make \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-$(uname -r) 2>/dev/null || apt-get install -y linux-perf && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential linux-tools-generic linux-tools-common && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/matrix_app

COPY workspace/matrix_app/compute.c /workspace/matrix_app/compute.c
COPY workspace/matrix_app/Makefile /workspace/matrix_app/Makefile

WORKDIR /workspace