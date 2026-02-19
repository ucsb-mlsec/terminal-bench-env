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
    gcc-mips-linux-gnu \
    binutils-mips-linux-gnu \
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y make gcc-mips-linux-gnu binutils-mips-linux-gnu && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/netmon/src

COPY netmon/src/netmon.c /workspace/netmon/src/netmon.c
COPY netmon/Makefile /workspace/netmon/Makefile

WORKDIR /workspace