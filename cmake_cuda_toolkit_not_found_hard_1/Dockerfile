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
    apt-get install -y cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/gpu_compute_project && \
    mkdir -p /opt/cuda-mock/bin && \
    mkdir -p /opt/cuda-mock/include && \
    mkdir -p /opt/cuda-mock/lib64

COPY workspace/gpu_compute_project/CMakeLists.txt /workspace/gpu_compute_project/CMakeLists.txt
COPY opt/cuda-mock/bin/nvcc /opt/cuda-mock/bin/nvcc
COPY opt/cuda-mock/include/cuda.h /opt/cuda-mock/include/cuda.h
COPY opt/cuda-mock/include/cuda_runtime.h /opt/cuda-mock/include/cuda_runtime.h
COPY opt/cuda-mock/lib64/libcudart.so /opt/cuda-mock/lib64/libcudart.so
COPY opt/cuda-mock/lib64/libcuda.so /opt/cuda-mock/lib64/libcuda.so

RUN chmod +x /opt/cuda-mock/bin/nvcc && \
    echo "11.4" > /opt/cuda-mock/version.txt

WORKDIR /workspace