FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/queue /home/agent/config /home/agent/data

COPY worker.py /home/agent/worker.py
COPY config/config.json /home/agent/config.json
COPY data/message_log.txt /home/agent/message_log.txt
COPY data/queue/msg_001.json /home/agent/queue/msg_001.json
COPY data/queue/msg_002.json /home/agent/queue/msg_002.json
COPY data/queue/msg_003.json /home/agent/queue/msg_003.json

WORKDIR /home/agent