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
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages numpy

RUN mkdir -p /workspace/tensors

COPY baseline_inference.py /workspace/baseline_inference.py
COPY model.py /workspace/model.py
COPY data/queue.txt /workspace/queue.txt
COPY data/baseline_time.txt /workspace/baseline_time.txt
COPY data/tensors/tensor_000.npy /workspace/tensors/tensor_000.npy
COPY data/tensors/tensor_001.npy /workspace/tensors/tensor_001.npy
COPY data/tensors/tensor_002.npy /workspace/tensors/tensor_002.npy
COPY data/tensors/tensor_003.npy /workspace/tensors/tensor_003.npy
COPY data/tensors/tensor_004.npy /workspace/tensors/tensor_004.npy
COPY data/tensors/tensors_005_to_099.txt /workspace/tensors/tensors_005_to_099.txt

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
