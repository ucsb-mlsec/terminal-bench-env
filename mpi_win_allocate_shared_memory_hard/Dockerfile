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
    libopenmpi-dev \
    openmpi-bin \
    time \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y build-essential libopenmpi-dev openmpi-bin && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/matrix_bench /workspace/data

COPY matrix_bench/matrix_benchmark.c /workspace/matrix_bench/matrix_benchmark.c
COPY matrix_bench/Makefile /workspace/matrix_bench/Makefile
COPY matrix_bench/generate_matrix.c /workspace/matrix_bench/generate_matrix.c
COPY matrix_bench/README.md /workspace/matrix_bench/README.md
COPY data/matrix_data.bin /workspace/data/matrix_data.bin

WORKDIR /workspace