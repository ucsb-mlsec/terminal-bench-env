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
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/matrix_calc/src

COPY workspace/matrix_calc/Makefile /workspace/matrix_calc/Makefile
COPY workspace/matrix_calc/src/matrix_mult.c /workspace/matrix_calc/src/matrix_mult.c
COPY workspace/matrix_calc/src/matrix_add.c /workspace/matrix_calc/src/matrix_add.c
COPY workspace/matrix_calc/src/matrix_transpose.c /workspace/matrix_calc/src/matrix_transpose.c

WORKDIR /workspace/matrix_calc

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
