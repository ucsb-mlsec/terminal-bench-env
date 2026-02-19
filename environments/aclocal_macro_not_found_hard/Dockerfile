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
    autoconf \
    automake \
    libtool \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y autoconf automake libtool pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/legacy-app

COPY legacy-app/main.c /workspace/legacy-app/main.c
COPY legacy-app/mathops.c /workspace/legacy-app/mathops.c
COPY legacy-app/mathops.h /workspace/legacy-app/mathops.h
COPY legacy-app/dbops.c /workspace/legacy-app/dbops.c
COPY legacy-app/dbops.h /workspace/legacy-app/dbops.h
COPY legacy-app/configure.ac /workspace/legacy-app/configure.ac
COPY legacy-app/Makefile.am /workspace/legacy-app/Makefile.am

WORKDIR /workspace