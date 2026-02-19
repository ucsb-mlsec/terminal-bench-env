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
    python3-cffi \
    libffi-dev \
    gcc && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages cffi

RUN mkdir -p /workspace/matrix_lib

COPY data/matrix_a.txt /workspace/matrix_lib/matrix_a.txt
COPY data/matrix_b.txt /workspace/matrix_lib/matrix_b.txt
COPY matrix_ops.c /workspace/matrix_lib/matrix_ops.c
COPY libmatrix.so /workspace/matrix_lib/libmatrix.so
COPY matrix_app.py /workspace/matrix_lib/matrix_app.py

WORKDIR /workspace