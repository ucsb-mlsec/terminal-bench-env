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
    apt-get install -y \
    vim \
    less \
    grep \
    jq \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/kafka/diagnostics /home/kafka

COPY data/consumer_app.log /home/kafka/diagnostics/consumer_app.log
COPY data/consumer_group_metrics.json /home/kafka/diagnostics/consumer_group_metrics.json
COPY data/consumer_config_transaction_processor.properties /home/kafka/diagnostics/consumer_config_transaction_processor.properties
COPY data/consumer_config_analytics_consumer.properties /home/kafka/diagnostics/consumer_config_analytics_consumer.properties
COPY data/consumer_config_notification_service.properties /home/kafka/diagnostics/consumer_config_notification_service.properties
COPY data/jvm_thread_dump.txt /home/kafka/diagnostics/jvm_thread_dump.txt
COPY data/jvm_heap_dump_summary.txt /home/kafka/diagnostics/jvm_heap_dump_summary.txt
COPY data/system_resources.log /home/kafka/diagnostics/system_resources.log
COPY data/network_connectivity_test.log /home/kafka/diagnostics/network_connectivity_test.log

WORKDIR /home/kafka

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
