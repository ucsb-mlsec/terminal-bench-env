FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages protobuf

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/service_a/protos /workspace/service_b/protos /workspace/service_c/protos

COPY service_a/protos/user.proto /workspace/service_a/protos/user.proto
COPY service_a/protos/order.proto /workspace/service_a/protos/order.proto
COPY service_b/protos/user.proto /workspace/service_b/protos/user.proto
COPY service_b/protos/order.proto /workspace/service_b/protos/order.proto
COPY service_c/protos/user.proto /workspace/service_c/protos/user.proto
COPY service_c/protos/order.proto /workspace/service_c/protos/order.proto

WORKDIR /workspace