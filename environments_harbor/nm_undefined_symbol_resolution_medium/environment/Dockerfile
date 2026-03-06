FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install build tools
RUN apt-get update && \
    apt-get install -y build-essential gcc make libc-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/app

# Copy the source files
COPY utils.h /workspace/app/utils.h
COPY utils.c /workspace/app/utils.c
COPY main.c /workspace/app/main.c
COPY Makefile /workspace/app/Makefile

WORKDIR /workspace/app

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
