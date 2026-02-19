FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/lora_configs

COPY data/lora_configs/experiment_1.json /workspace/lora_configs/experiment_1.json
COPY data/lora_configs/experiment_2.json /workspace/lora_configs/experiment_2.json
COPY data/lora_configs/experiment_3.json /workspace/lora_configs/experiment_3.json
COPY data/lora_configs/experiment_4.json /workspace/lora_configs/experiment_4.json
COPY data/lora_configs/experiment_5.json /workspace/lora_configs/experiment_5.json
COPY data/lora_configs/experiment_6.json /workspace/lora_configs/experiment_6.json
COPY data/lora_configs/experiment_7.json /workspace/lora_configs/experiment_7.json
COPY data/lora_configs/experiment_8.json /workspace/lora_configs/experiment_8.json
COPY data/lora_configs/experiment_9.json /workspace/lora_configs/experiment_9.json
COPY data/lora_configs/experiment_10.json /workspace/lora_configs/experiment_10.json
COPY data/lora_configs/experiment_11.json /workspace/lora_configs/experiment_11.json
COPY data/lora_configs/experiment_12.json /workspace/lora_configs/experiment_12.json
COPY data/lora_configs/experiment_13.json /workspace/lora_configs/experiment_13.json
COPY data/lora_configs/experiment_14.json /workspace/lora_configs/experiment_14.json
COPY data/lora_configs/experiment_15.json /workspace/lora_configs/experiment_15.json

WORKDIR /workspace