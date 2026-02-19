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

RUN mkdir -p /tmp/project/auth /tmp/project/network /tmp/project/storage /tmp/project/main

COPY auth/Makefile /tmp/project/auth/Makefile
COPY auth/auth.c /tmp/project/auth/auth.c
COPY network/Makefile /tmp/project/network/Makefile
COPY network/network.c /tmp/project/network/network.c
COPY storage/Makefile /tmp/project/storage/Makefile
COPY storage/storage.c /tmp/project/storage/storage.c
COPY main/Makefile /tmp/project/main/Makefile
COPY main/main.c /tmp/project/main/main.c

WORKDIR /tmp/project