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

RUN mkdir -p /workspace/data /workspace/config

COPY user-service.xsd /workspace/user-service.xsd
COPY order-service.xsd /workspace/order-service.xsd
COPY inventory-service.xsd /workspace/inventory-service.xsd
COPY data/user-message.xml /workspace/data/user-message.xml
COPY data/order-message.xml /workspace/data/order-message.xml
COPY data/inventory-message.xml /workspace/data/inventory-message.xml
COPY config/validation-config.xml /workspace/config/validation-config.xml
COPY data/error-logs.txt /workspace/data/error-logs.txt

WORKDIR /workspace