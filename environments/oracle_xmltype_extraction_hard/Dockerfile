FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data/orders /workspace/data/customers /workspace/data/shipments

COPY data/orders/order_ORD-12345.xml /workspace/data/orders/order_ORD-12345.xml
COPY data/orders/order_ORD-12346.xml /workspace/data/orders/order_ORD-12346.xml
COPY data/orders/order_ORD-TEST-999.xml /workspace/data/orders/order_ORD-TEST-999.xml
COPY data/orders/order_ORD-12347.xml /workspace/data/orders/order_ORD-12347.xml
COPY data/orders/order_ORD-12349.xml /workspace/data/orders/order_ORD-12349.xml
COPY data/customers/customer_CUST-001.xml /workspace/data/customers/customer_CUST-001.xml
COPY data/customers/customer_CUST-002.xml /workspace/data/customers/customer_CUST-002.xml
COPY data/customers/customer_CUST-003.xml /workspace/data/customers/customer_CUST-003.xml
COPY data/customers/customer_CUST-004.xml /workspace/data/customers/customer_CUST-004.xml
COPY data/shipments/shipment_ORD-12345.xml /workspace/data/shipments/shipment_ORD-12345.xml
COPY data/shipments/shipment_ORD-12346.xml /workspace/data/shipments/shipment_ORD-12346.xml
COPY data/shipments/shipment_ORD-TEST-999.xml /workspace/data/shipments/shipment_ORD-TEST-999.xml
COPY data/shipments/shipment_ORD-12348.xml /workspace/data/shipments/shipment_ORD-12348.xml

WORKDIR /workspace