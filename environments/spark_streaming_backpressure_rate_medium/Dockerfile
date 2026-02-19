FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/streaming-configs /tmp

COPY config/prod-streaming.conf /opt/streaming-configs/prod-streaming.conf
COPY config/staging-v2.conf /opt/streaming-configs/staging-v2.conf
COPY config/dev-streaming.conf /opt/streaming-configs/dev-streaming.conf
COPY config/test-basic.conf /opt/streaming-configs/test-basic.conf
COPY config/prod-kafka-streaming.conf /opt/streaming-configs/prod-kafka-streaming.conf
COPY config/legacy-stream.conf /opt/streaming-configs/legacy-stream.conf
COPY config/staging-incomplete.conf /opt/streaming-configs/staging-incomplete.conf

WORKDIR /workspace