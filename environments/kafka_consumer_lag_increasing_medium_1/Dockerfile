FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/kafka_diagnostics/data /home/agent/kafka_diagnostics/config

COPY data/consumer_offsets.log /home/agent/kafka_diagnostics/consumer_offsets.log
COPY data/topic_metrics.log /home/agent/kafka_diagnostics/topic_metrics.log
COPY config/consumer_config.properties /home/agent/kafka_diagnostics/consumer_config.properties
COPY data/processing_times.csv /home/agent/kafka_diagnostics/processing_times.csv
COPY data/jvm_metrics.log /home/agent/kafka_diagnostics/jvm_metrics.log

WORKDIR /home/agent