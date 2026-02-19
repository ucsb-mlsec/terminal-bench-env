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
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/cuda_src

COPY cuda_src/tree_traverse.cu /workspace/cuda_src/tree_traverse.cu
COPY cuda_src/graph_process.cu /workspace/cuda_src/graph_process.cu
COPY cuda_src/hierarchical_scan.cu /workspace/cuda_src/hierarchical_scan.cu
COPY cuda_src/recursive_reduce.cu /workspace/cuda_src/recursive_reduce.cu
COPY cuda_src/adaptive_traverse.cu /workspace/cuda_src/adaptive_traverse.cu
COPY cuda_src/parallel_search.cu /workspace/cuda_src/parallel_search.cu

WORKDIR /workspace