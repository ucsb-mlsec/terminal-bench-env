FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data /workspace/config

COPY data/kafka_metrics.json /workspace/kafka_metrics.json
COPY data/consumer_app.log /workspace/consumer_app.log
COPY config/consumer_config.properties /workspace/consumer_config.properties
COPY data/jvm_metrics.txt /workspace/jvm_metrics.txt

WORKDIR /workspace