FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /cluster/checkpoints/job_847392 \
    /cluster/logs/job_847392 \
    /cluster/metadata/job_847392 \
    /solution

COPY data/checkpoints/job_847392/checkpoint_2024-01-12T08:15:23.chk /cluster/checkpoints/job_847392/
COPY data/checkpoints/job_847392/checkpoint_2024-01-13T16:42:11.chk /cluster/checkpoints/job_847392/
COPY data/checkpoints/job_847392/checkpoint_2024-01-15T14:23:17.chk /cluster/checkpoints/job_847392/
COPY data/checkpoints/job_847392/checkpoint_2024-01-15T18:51:04.chk.tmp /cluster/checkpoints/job_847392/
COPY data/checkpoints/job_847392/checkpoint_2024-01-10T05:23:11.chk /cluster/checkpoints/job_847392/
COPY data/checkpoints/job_847392/checkpoint_2024-01-14T22:15:38.chk /cluster/checkpoints/job_847392/

COPY data/logs/job_847392/slurm-847392.out /cluster/logs/job_847392/
COPY data/logs/job_847392/slurm-847392.err /cluster/logs/job_847392/

COPY data/metadata/job_847392/job_config.json /cluster/metadata/job_847392/
COPY data/metadata/job_847392/job_state.txt /cluster/metadata/job_847392/
COPY data/metadata/job_847392/checkpoint_manifest.log /cluster/metadata/job_847392/

WORKDIR /cluster