FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/mappings /workspace/rejected_docs

COPY data/mappings/products_mapping.json /workspace/mappings/products_mapping.json
COPY data/mappings/users_mapping.json /workspace/mappings/users_mapping.json
COPY data/mappings/orders_mapping.json /workspace/mappings/orders_mapping.json
COPY data/rejected_docs/rejected_001.json /workspace/rejected_docs/rejected_001.json
COPY data/rejected_docs/rejected_002.json /workspace/rejected_docs/rejected_002.json
COPY data/rejected_docs/rejected_004.json /workspace/rejected_docs/rejected_004.json
COPY data/rejected_docs/rejected_005.json /workspace/rejected_docs/rejected_005.json

WORKDIR /workspace