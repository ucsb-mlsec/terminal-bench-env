FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/pulsar_simulation/orders \
    /workspace/pulsar_simulation/inventory \
    /workspace/pulsar_simulation/notifications

COPY data/pulsar_simulation/orders/msg_295.json /workspace/pulsar_simulation/orders/msg_295.json
COPY data/pulsar_simulation/orders/msg_298.json /workspace/pulsar_simulation/orders/msg_298.json
COPY data/pulsar_simulation/orders/msg_301.json /workspace/pulsar_simulation/orders/msg_301.json
COPY data/pulsar_simulation/orders/msg_305.json /workspace/pulsar_simulation/orders/msg_305.json
COPY data/pulsar_simulation/orders/msg_308.json /workspace/pulsar_simulation/orders/msg_308.json
COPY data/pulsar_simulation/orders/msg_312.json /workspace/pulsar_simulation/orders/msg_312.json
COPY data/pulsar_simulation/orders/subscription.json /workspace/pulsar_simulation/orders/subscription.json

COPY data/pulsar_simulation/inventory/msg_270.json /workspace/pulsar_simulation/inventory/msg_270.json
COPY data/pulsar_simulation/inventory/msg_273.json /workspace/pulsar_simulation/inventory/msg_273.json
COPY data/pulsar_simulation/inventory/msg_278.json /workspace/pulsar_simulation/inventory/msg_278.json
COPY data/pulsar_simulation/inventory/msg_281.json /workspace/pulsar_simulation/inventory/msg_281.json
COPY data/pulsar_simulation/inventory/msg_285.json /workspace/pulsar_simulation/inventory/msg_285.json
COPY data/pulsar_simulation/inventory/msg_289.json /workspace/pulsar_simulation/inventory/msg_289.json
COPY data/pulsar_simulation/inventory/subscription.json /workspace/pulsar_simulation/inventory/subscription.json

COPY data/pulsar_simulation/notifications/msg_230.json /workspace/pulsar_simulation/notifications/msg_230.json
COPY data/pulsar_simulation/notifications/msg_234.json /workspace/pulsar_simulation/notifications/msg_234.json
COPY data/pulsar_simulation/notifications/msg_239.json /workspace/pulsar_simulation/notifications/msg_239.json
COPY data/pulsar_simulation/notifications/msg_242.json /workspace/pulsar_simulation/notifications/msg_242.json
COPY data/pulsar_simulation/notifications/msg_246.json /workspace/pulsar_simulation/notifications/msg_246.json
COPY data/pulsar_simulation/notifications/subscription.json /workspace/pulsar_simulation/notifications/subscription.json

WORKDIR /workspace