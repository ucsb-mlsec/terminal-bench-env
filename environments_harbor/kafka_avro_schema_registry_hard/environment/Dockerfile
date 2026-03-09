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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages avro-python3

RUN mkdir -p /workspace/schemas /workspace/data /workspace/solution

COPY data/schemas/user_schema.avsc /workspace/schemas/user_schema.avsc
COPY data/schemas/order_schema.avsc /workspace/schemas/order_schema.avsc
COPY data/schemas/product_schema.avsc /workspace/schemas/product_schema.avsc
COPY data/schemas/payment_schema.avsc /workspace/schemas/payment_schema.avsc
COPY data/schema_mapping.txt /workspace/schema_mapping.txt
COPY data/schema_mapping.txt /workspace/data/schema_mapping.txt
COPY data/user_events.json /workspace/data/user_events.json
COPY data/order_data.json /workspace/data/order_data.json
COPY data/product_catalog.json /workspace/data/product_catalog.json
COPY data/inventory_updates.json /workspace/data/inventory_updates.json
COPY data/payment_transactions.json /workspace/data/payment_transactions.json
COPY data/customer_profiles.json /workspace/data/customer_profiles.json
COPY data/shipping_notifications.json /workspace/data/shipping_notifications.json
COPY data/analytics_events.json /workspace/data/analytics_events.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
