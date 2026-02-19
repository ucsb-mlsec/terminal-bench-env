FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /cluster_configs /solution

COPY cluster_configs/slurm.conf /cluster_configs/slurm.conf
COPY cluster_configs/gres.conf /cluster_configs/gres.conf
COPY cluster_configs/node_features.conf /cluster_configs/node_features.conf

WORKDIR /workspace