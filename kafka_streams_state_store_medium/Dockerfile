FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/kafka-app/config \
    /workspace/kafka-app/logs \
    /workspace/kafka-app/docs

COPY config/streams.properties /workspace/kafka-app/config/streams.properties
COPY StateStoreTopology.java /workspace/kafka-app/StateStoreTopology.java
COPY data/application.log /workspace/kafka-app/logs/application.log
COPY data/state-store-requirements.txt /workspace/kafka-app/docs/state-store-requirements.txt

WORKDIR /workspace