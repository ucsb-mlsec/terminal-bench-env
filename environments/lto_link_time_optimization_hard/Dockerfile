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

RUN mkdir -p /workspace/matrixlib

COPY matrixlib/matrix_ops.c /workspace/matrixlib/matrix_ops.c
COPY matrixlib/matrix_utils.c /workspace/matrixlib/matrix_utils.c
COPY matrixlib/matrix_ops.h /workspace/matrixlib/matrix_ops.h
COPY matrixlib/matrix_bench.c /workspace/matrixlib/matrix_bench.c
COPY matrixlib/Makefile /workspace/matrixlib/Makefile
COPY matrixlib/test_data.txt /workspace/matrixlib/test_data.txt

WORKDIR /workspace