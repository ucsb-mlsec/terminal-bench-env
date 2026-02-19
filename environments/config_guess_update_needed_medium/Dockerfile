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
    autoconf \
    automake \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y autoconf automake build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/legacy-app/build-aux

COPY legacy-app/configure /workspace/legacy-app/configure
COPY legacy-app/build-aux/config.guess /workspace/legacy-app/build-aux/config.guess
COPY legacy-app/build-aux/config.sub /workspace/legacy-app/build-aux/config.sub
COPY legacy-app/Makefile.in /workspace/legacy-app/Makefile.in
COPY legacy-app/main.c /workspace/legacy-app/main.c
COPY legacy-app/configure.ac /workspace/legacy-app/configure.ac

RUN chmod +x /workspace/legacy-app/configure && \
    chmod +x /workspace/legacy-app/build-aux/config.guess && \
    chmod +x /workspace/legacy-app/build-aux/config.sub

WORKDIR /workspace