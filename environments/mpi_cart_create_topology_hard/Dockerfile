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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /workspace/job_configs

COPY data/job_configs/job_001.yaml /workspace/job_configs/job_001.yaml
COPY data/job_configs/job_002.yaml /workspace/job_configs/job_002.yaml
COPY data/job_configs/job_003.yaml /workspace/job_configs/job_003.yaml
COPY data/job_configs/job_004.yaml /workspace/job_configs/job_004.yaml
COPY data/job_configs/job_005.yaml /workspace/job_configs/job_005.yaml
COPY data/job_configs/job_006.yaml /workspace/job_configs/job_006.yaml
COPY data/job_configs/job_007.yaml /workspace/job_configs/job_007.yaml
COPY data/job_configs/job_008.yaml /workspace/job_configs/job_008.yaml
COPY data/job_configs/job_009.yaml /workspace/job_configs/job_009.yaml
COPY data/job_configs/job_010.yaml /workspace/job_configs/job_010.yaml
COPY data/job_configs/job_011.yaml /workspace/job_configs/job_011.yaml
COPY data/job_configs/job_012.yaml /workspace/job_configs/job_012.yaml
COPY data/job_configs/job_013.yaml /workspace/job_configs/job_013.yaml
COPY data/job_configs/job_014.yaml /workspace/job_configs/job_014.yaml
COPY data/job_configs/job_015.yaml /workspace/job_configs/job_015.yaml

WORKDIR /workspace