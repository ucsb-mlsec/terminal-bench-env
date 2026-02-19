FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/system_snapshot/system_config

COPY data/gpu_info.txt /workspace/system_snapshot/gpu_info.txt
COPY data/process_list.txt /workspace/system_snapshot/process_list.txt
COPY data/service_status.txt /workspace/system_snapshot/service_status.txt
COPY data/environment_vars.txt /workspace/system_snapshot/environment_vars.txt
COPY data/system_config/gpu.conf /workspace/system_snapshot/system_config/gpu.conf
COPY data/system_config/cuda.conf /workspace/system_snapshot/system_config/cuda.conf

WORKDIR /workspace