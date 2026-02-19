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
    gcc \
    make \
    binutils \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc g++ make binutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/libmath /workspace/tests

COPY libmath/math_lib.c /workspace/libmath/math_lib.c
COPY libmath/Makefile /workspace/libmath/Makefile
COPY libmath/libmath.map /workspace/libmath/libmath.map
COPY tests/legacy_app.c /workspace/tests/legacy_app.c
COPY tests/modern_app.c /workspace/tests/modern_app.c

WORKDIR /workspace