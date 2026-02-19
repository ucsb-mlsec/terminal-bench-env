FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/queue/messages /tmp

COPY data/msg-001.json /var/queue/messages/msg-001.json
COPY data/msg-002.json /var/queue/messages/msg-002.json
COPY data/msg-003.json /var/queue/messages/msg-003.json
COPY data/msg-004.json /var/queue/messages/msg-004.json
COPY data/msg-005.json /var/queue/messages/msg-005.json
COPY process_queue.py /workspace/process_queue.py

WORKDIR /workspace