FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cuda_samples

COPY cuda_samples/processor_a.cu /workspace/cuda_samples/processor_a.cu
COPY cuda_samples/processor_b.cu /workspace/cuda_samples/processor_b.cu
COPY cuda_samples/processor_c.cu /workspace/cuda_samples/processor_c.cu
COPY cuda_samples/kernel_utils.cu /workspace/cuda_samples/kernel_utils.cu
COPY cuda_samples/memory_manager.cu /workspace/cuda_samples/memory_manager.cu

WORKDIR /workspace