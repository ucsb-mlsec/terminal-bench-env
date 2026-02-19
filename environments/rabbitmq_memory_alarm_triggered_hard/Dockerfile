FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/rabbitmq_diagnostics /tmp

COPY data/memory_status.json /tmp/rabbitmq_diagnostics/memory_status.json
COPY data/queue_details.json /tmp/rabbitmq_diagnostics/queue_details.json
COPY data/connection_info.json /tmp/rabbitmq_diagnostics/connection_info.json
COPY data/cluster_config.txt /tmp/rabbitmq_diagnostics/cluster_config.txt
COPY data/alarm_log.txt /tmp/rabbitmq_diagnostics/alarm_log.txt

WORKDIR /workspace