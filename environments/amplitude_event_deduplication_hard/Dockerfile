FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/events /solution

COPY data/events/events_batch_001.json /data/events/events_batch_001.json
COPY data/events/events_batch_002.json /data/events/events_batch_002.json
COPY data/events/events_batch_003.json /data/events/events_batch_003.json
COPY data/events/events_batch_004.json /data/events/events_batch_004.json
COPY data/events/events_batch_005.json /data/events/events_batch_005.json

WORKDIR /workspace