FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages flask requests

RUN mkdir -p /home/user/sample_webhooks /tmp

COPY webhook_service.py /home/user/webhook_service.py
COPY data/sample_webhooks/payment1.json /home/user/sample_webhooks/payment1.json
COPY data/sample_webhooks/payment2.json /home/user/sample_webhooks/payment2.json
COPY data/sample_webhooks/event1.json /home/user/sample_webhooks/event1.json
COPY data/sample_webhooks/event2.json /home/user/sample_webhooks/event2.json
COPY data/sample_webhooks/alert1.json /home/user/sample_webhooks/alert1.json
COPY data/sample_webhooks/alert2.json /home/user/sample_webhooks/alert2.json

WORKDIR /home/user