FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /app/dlq /app/processor /app/test_messages

COPY data/failed_messages.json /app/dlq/failed_messages.json
COPY processor/order_processor.py /app/processor/order_processor.py
COPY processor/validator.py /app/processor/validator.py
COPY test_messages/valid_order_1.json /app/test_messages/valid_order_1.json

WORKDIR /app