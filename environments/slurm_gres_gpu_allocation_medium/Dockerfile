FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/job_scripts

COPY cluster_config.txt /workspace/cluster_config.txt
COPY job_scripts/train_basic.sh /workspace/job_scripts/train_basic.sh
COPY job_scripts/train_multi.sh /workspace/job_scripts/train_multi.sh
COPY job_scripts/train_v100.sh /workspace/job_scripts/train_v100.sh
COPY job_scripts/inference_a100.sh /workspace/job_scripts/inference_a100.sh
COPY job_scripts/distributed_train.sh /workspace/job_scripts/distributed_train.sh

WORKDIR /workspace