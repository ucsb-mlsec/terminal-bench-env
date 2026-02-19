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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask

RUN mkdir -p /opt/webhooks /var/log/webhooks

COPY config/secrets.conf /opt/webhooks/secrets.conf
COPY payment_webhook.py /opt/webhooks/payment_webhook.py
COPY notification_webhook.py /opt/webhooks/notification_webhook.py
COPY sync_webhook.py /opt/webhooks/sync_webhook.py
COPY data/webhook_logs.json /var/log/webhooks/webhook_logs.json

RUN chmod +r /opt/webhooks/secrets.conf && \
    chmod +r /opt/webhooks/*.py && \
    chmod +r /var/log/webhooks/webhook_logs.json

WORKDIR /opt/webhooks