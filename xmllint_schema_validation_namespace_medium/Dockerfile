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
    apt-get install -y libxml2-utils && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages lxml

RUN mkdir -p /workspace/data /workspace/schemas

COPY data/customer_orders.xml /workspace/data/customer_orders.xml
COPY data/inventory_update.xml /workspace/data/inventory_update.xml
COPY data/shipping_manifest.xml /workspace/data/shipping_manifest.xml
COPY schemas/customer_orders.xsd /workspace/schemas/customer_orders.xsd
COPY schemas/inventory_update.xsd /workspace/schemas/inventory_update.xsd
COPY schemas/shipping_manifest.xsd /workspace/schemas/shipping_manifest.xsd

WORKDIR /workspace