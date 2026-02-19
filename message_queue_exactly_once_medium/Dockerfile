FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/message_processor \
    /workspace/logs \
    /workspace/output \
    /workspace/test_queue

COPY message_processor/__init__.py /workspace/message_processor/__init__.py
COPY message_processor/processor.py /workspace/message_processor/processor.py
COPY data/message_history.log /workspace/logs/message_history.log
COPY data/messages.json /workspace/test_queue/messages.json

WORKDIR /workspace