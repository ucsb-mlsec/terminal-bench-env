FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace /tmp

COPY image_processing.cu /workspace/image_processing.cu
COPY data_transfer.cu /workspace/data_transfer.cu
COPY matrix_ops.cu /workspace/matrix_ops.cu
COPY memory_utils.h /workspace/memory_utils.h
COPY vector_add.cu /workspace/vector_add.cu
COPY stream_processing.cu /workspace/stream_processing.cu

WORKDIR /workspace