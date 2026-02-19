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
    apt-get install -y build-essential make gcc && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y binutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/project/src && \
    mkdir -p /workspace/project/libs && \
    mkdir -p /workspace/project/build

COPY workspace/project/src/main.c /workspace/project/src/main.c
COPY workspace/project/libs/libmath.a /workspace/project/libs/libmath.a
COPY workspace/project/libs/libutils.a /workspace/project/libs/libutils.a
COPY workspace/project/Makefile /workspace/project/Makefile

WORKDIR /workspace