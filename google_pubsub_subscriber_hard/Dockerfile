FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/pubsub/topics/user-events \
    /var/pubsub/topics/system-metrics \
    /var/pubsub/topics/error-logs \
    /var/pubsub/processed \
    /var/pubsub/dead-letter \
    /home/agent \
    /app/data

COPY data/user-events-001.json /var/pubsub/topics/user-events/user-events-001.json
COPY data/user-events-002.json /var/pubsub/topics/user-events/user-events-002.json
COPY data/user-events-003.json /var/pubsub/topics/user-events/user-events-003.json
COPY data/user-events-004-malformed.json /var/pubsub/topics/user-events/user-events-004-malformed.json
COPY data/user-events-005.json /var/pubsub/topics/user-events/user-events-005.json
COPY data/system-metrics-001.json /var/pubsub/topics/system-metrics/system-metrics-001.json
COPY data/system-metrics-002.json /var/pubsub/topics/system-metrics/system-metrics-002.json
COPY data/system-metrics-003-malformed.json /var/pubsub/topics/system-metrics/system-metrics-003-malformed.json
COPY data/system-metrics-004.json /var/pubsub/topics/system-metrics/system-metrics-004.json
COPY data/error-logs-001.json /var/pubsub/topics/error-logs/error-logs-001.json
COPY data/error-logs-003-malformed.json /var/pubsub/topics/error-logs/error-logs-003-malformed.json
COPY broken_subscriber.py /app/broken_subscriber.py

WORKDIR /app