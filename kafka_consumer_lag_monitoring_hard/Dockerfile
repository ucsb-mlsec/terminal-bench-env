FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /kafka-diagnostics /root/solution

COPY data/consumer_group_analytics_processors.json /kafka-diagnostics/consumer_group_analytics_processors.json
COPY data/consumer_group_data_exporters.json /kafka-diagnostics/consumer_group_data_exporters.json
COPY data/consumer_group_realtime_alerts.json /kafka-diagnostics/consumer_group_realtime_alerts.json
COPY data/topic_metadata.json /kafka-diagnostics/topic_metadata.json
COPY data/consumer_configurations.json /kafka-diagnostics/consumer_configurations.json
COPY data/lag_history.json /kafka-diagnostics/lag_history.json

WORKDIR /root