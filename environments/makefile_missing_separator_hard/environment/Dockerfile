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

RUN mkdir -p /workspace/src /workspace/lib/crypto /workspace/lib/network

COPY Makefile /workspace/Makefile
COPY src/main.c /workspace/src/main.c
COPY lib/crypto/Makefile /workspace/lib/crypto/Makefile
COPY lib/crypto/crypto.c /workspace/lib/crypto/crypto.c
COPY lib/crypto/crypto.h /workspace/lib/crypto/crypto.h
COPY lib/network/Makefile /workspace/lib/network/Makefile
COPY lib/network/network.c /workspace/lib/network/network.c
COPY lib/network/network.h /workspace/lib/network/network.h

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
