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
    python3-pip \
    python3-yaml && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages dask[distributed]

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /workspace/dask_configs

COPY config/dask_config_1.yaml /workspace/dask_configs/dask_config_1.yaml
COPY config/dask_config_2.yaml /workspace/dask_configs/dask_config_2.yaml
COPY config/dask_config_3.yaml /workspace/dask_configs/dask_config_3.yaml
COPY config/dask_config_4.yaml /workspace/dask_configs/dask_config_4.yaml
COPY config/dask_config_5.yaml /workspace/dask_configs/dask_config_5.yaml

WORKDIR /workspace