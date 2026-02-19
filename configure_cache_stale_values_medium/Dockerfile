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
    autoconf \
    automake \
    libsqlite3-dev \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make autoconf sqlite3 libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/myapp

COPY configure.ac /workspace/myapp/configure.ac
COPY configure /workspace/myapp/configure
COPY Makefile.in /workspace/myapp/Makefile.in
COPY main.c /workspace/myapp/main.c
COPY config.cache /workspace/myapp/config.cache

RUN chmod +x /workspace/myapp/configure

WORKDIR /workspace/myapp