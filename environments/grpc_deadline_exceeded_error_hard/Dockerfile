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

RUN pip3 install --no-cache-dir --break-system-packages grpcio grpcio-tools protobuf

RUN mkdir -p /home/agent/order_service

COPY order_service/service.proto /home/agent/order_service/service.proto
COPY order_service/order_pb2_grpc.py /home/agent/order_service/order_pb2_grpc.py
COPY order_service/server.py /home/agent/order_service/server.py
COPY order_service/test_client.py /home/agent/order_service/test_client.py

WORKDIR /home/agent/order_service