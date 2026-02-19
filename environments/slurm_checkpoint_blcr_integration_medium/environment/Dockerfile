FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y python3 && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/checkpoints/job_5241 \
    /workspace/jobs \
    /workspace/config \
    /workspace/solution \
    /etc/slurm-llnl

COPY data/checkpoint_2024-01-15_14-30-22.json /workspace/checkpoints/job_5241/
COPY data/checkpoint_2024-01-15_14-40-15.json /workspace/checkpoints/job_5241/
COPY data/checkpoint_2024-01-15_14-50-33.json /workspace/checkpoints/job_5241/
COPY data/checkpoint_2024-01-15_15-00-47.json /workspace/checkpoints/job_5241/
COPY data/checkpoint_2024-01-15_15-05-12.json /workspace/checkpoints/job_5241/
COPY jobs/process_batch.sh /workspace/jobs/
COPY config/app.conf /workspace/config/

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
