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
    gcc \
    make \
    binutils \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc binutils make libc6-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/firmware

COPY firmware/main.c /workspace/firmware/main.c
COPY firmware/fast_code.c /workspace/firmware/fast_code.c
COPY firmware/slow_code.c /workspace/firmware/slow_code.c
COPY firmware/util.c /workspace/firmware/util.c
COPY firmware/Makefile /workspace/firmware/Makefile
COPY firmware/linker.ld /workspace/firmware/linker.ld

WORKDIR /workspace