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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyyaml

RUN mkdir -p /home/user /data /config

COPY data/task_config_cold.json /data/task_config_cold.json
COPY data/task_config_warm_uswest2.json /data/task_config_warm_uswest2.json
COPY data/task_config_cold_uswest2.json /data/task_config_cold_uswest2.json
COPY data/execution_log_20240115.json /data/execution_log_20240115.json
COPY data/execution_log_20240117.json /data/execution_log_20240117.json
COPY data/location_status.json /data/location_status.json
COPY data/active_executions.json /data/active_executions.json
COPY config/orchestrator_config.yaml /config/orchestrator_config.yaml

WORKDIR /home/user