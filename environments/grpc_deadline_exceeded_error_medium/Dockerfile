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

RUN pip3 install --no-cache-dir --break-system-packages grpcio grpcio-tools protobuf pyyaml

RUN mkdir -p /workspace/order_system

COPY config/config.yaml /workspace/order_system/config.yaml
COPY test_orders.py /workspace/order_system/test_orders.py
COPY server.py /workspace/order_system/server.py
COPY client.py /workspace/order_system/client.py

WORKDIR /workspace