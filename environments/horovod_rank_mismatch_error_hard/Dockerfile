FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/config /workspace/logs /workspace/env /workspace/data

COPY config/cluster.yaml /workspace/config/cluster.yaml
COPY config/worker1.yaml /workspace/config/worker1.yaml
COPY config/worker2.yaml /workspace/config/worker2.yaml
COPY config/worker3.yaml /workspace/config/worker3.yaml
COPY config/worker4.yaml /workspace/config/worker4.yaml
COPY data/training_run_1.log /workspace/logs/training_run_1.log
COPY data/training_run_2.log /workspace/logs/training_run_2.log
COPY data/training_run_3.log /workspace/logs/training_run_3.log
COPY env/worker_env.txt /workspace/env/worker_env.txt
COPY train_distributed.py /workspace/train_distributed.py

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

WORKDIR /workspace