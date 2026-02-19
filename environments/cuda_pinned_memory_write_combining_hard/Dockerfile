FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cuda_specs

COPY cuda_specs/sequential_pattern.txt /workspace/cuda_specs/sequential_pattern.txt
COPY cuda_specs/strided_pattern.txt /workspace/cuda_specs/strided_pattern.txt
COPY cuda_specs/random_pattern.txt /workspace/cuda_specs/random_pattern.txt
COPY cuda_specs/cuda_memory_ref.txt /workspace/cuda_specs/cuda_memory_ref.txt

WORKDIR /workspace