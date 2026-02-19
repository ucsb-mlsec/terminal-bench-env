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
    m4 \
    make \
    gcc \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y autoconf automake libtool m4 make gcc g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/project/m4

COPY project/configure.ac /workspace/project/configure.ac
COPY project/m4/custom.m4 /workspace/project/m4/custom.m4
COPY project/Makefile.am /workspace/project/Makefile.am
COPY project/main.c /workspace/project/main.c

WORKDIR /workspace/project

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
