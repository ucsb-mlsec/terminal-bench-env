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
    apt-get install -y build-essential gcc make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y libc6-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY Makefile /workspace/Makefile
COPY add.c /workspace/add.c
COPY subtract.c /workspace/subtract.c
COPY multiply.c /workspace/multiply.c
COPY divide.c /workspace/divide.c
COPY mathutils.h /workspace/mathutils.h

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
