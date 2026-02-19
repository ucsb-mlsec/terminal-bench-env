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
    build-essential \
    libtool \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y autoconf automake gcc make libtool && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/legacy-calc/src

COPY configure.ac /workspace/legacy-calc/configure.ac
COPY Makefile.am /workspace/legacy-calc/Makefile.am
COPY src/Makefile.am /workspace/legacy-calc/src/Makefile.am
COPY src/calculator.c /workspace/legacy-calc/src/calculator.c

WORKDIR /workspace/legacy-calc