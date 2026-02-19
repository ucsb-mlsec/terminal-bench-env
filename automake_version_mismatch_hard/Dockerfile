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
    automake \
    autoconf \
    libtool \
    make \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y automake autoconf libtool make gcc g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/legacy-calc

COPY workspace/legacy-calc/calc.c /workspace/legacy-calc/calc.c
COPY workspace/legacy-calc/configure.ac /workspace/legacy-calc/configure.ac
COPY workspace/legacy-calc/Makefile.am /workspace/legacy-calc/Makefile.am
COPY workspace/legacy-calc/Makefile.in /workspace/legacy-calc/Makefile.in
COPY workspace/legacy-calc/aclocal.m4 /workspace/legacy-calc/aclocal.m4
COPY workspace/legacy-calc/config.h.in /workspace/legacy-calc/config.h.in

WORKDIR /workspace/legacy-calc