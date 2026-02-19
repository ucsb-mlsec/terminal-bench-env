FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /tmp/dlq_events /tmp

COPY data/dlq_events/event_001.json /tmp/dlq_events/event_001.json
COPY data/dlq_events/event_002.json /tmp/dlq_events/event_002.json
COPY data/dlq_events/event_003.json /tmp/dlq_events/event_003.json
COPY data/dlq_events/event_004.json /tmp/dlq_events/event_004.json
COPY data/dlq_events/event_006.json /tmp/dlq_events/event_006.json
COPY data/dlq_events/event_007.json /tmp/dlq_events/event_007.json
COPY data/dlq_events/event_008.json /tmp/dlq_events/event_008.json
COPY data/dlq_events/event_009.json /tmp/dlq_events/event_009.json
COPY data/dlq_events/event_010.json /tmp/dlq_events/event_010.json

WORKDIR /tmp

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
