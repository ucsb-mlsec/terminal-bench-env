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
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyarrow pandas

RUN mkdir -p /data/spark_metadata/warehouse/customer_orders && \
    mkdir -p /data/spark_metadata/warehouse/product_catalog && \
    mkdir -p /data/spark_metadata/warehouse/user_sessions && \
    mkdir -p /data/spark_metadata/warehouse/transaction_logs && \
    mkdir -p /data/spark_metadata/warehouse/inventory_snapshot && \
    mkdir -p /data/spark_metadata/warehouse/sales_summary && \
    mkdir -p /data/spark_metadata/warehouse/click_events && \
    mkdir -p /data/spark_metadata/warehouse/user_profiles && \
    mkdir -p /data/spark_metadata/warehouse/payment_records && \
    mkdir -p /tmp

COPY data/spark_metadata/tables.json /data/spark_metadata/tables.json
COPY data/spark_metadata/statistics.json /data/spark_metadata/statistics.json
COPY data/spark_metadata/warehouse/customer_orders/part-00000.parquet /data/spark_metadata/warehouse/customer_orders/part-00000.parquet
COPY data/spark_metadata/warehouse/customer_orders/part-00001.parquet /data/spark_metadata/warehouse/customer_orders/part-00001.parquet
COPY data/spark_metadata/warehouse/product_catalog/data.parquet /data/spark_metadata/warehouse/product_catalog/data.parquet
COPY data/spark_metadata/warehouse/user_sessions/sessions.parquet /data/spark_metadata/warehouse/user_sessions/sessions.parquet
COPY data/spark_metadata/warehouse/transaction_logs/part-00000.parquet /data/spark_metadata/warehouse/transaction_logs/part-00000.parquet
COPY data/spark_metadata/warehouse/transaction_logs/part-00001.parquet /data/spark_metadata/warehouse/transaction_logs/part-00001.parquet
COPY data/spark_metadata/warehouse/inventory_snapshot/snapshot.parquet /data/spark_metadata/warehouse/inventory_snapshot/snapshot.parquet
COPY data/spark_metadata/warehouse/sales_summary/summary.parquet /data/spark_metadata/warehouse/sales_summary/summary.parquet
COPY data/spark_metadata/warehouse/click_events/events-001.parquet /data/spark_metadata/warehouse/click_events/events-001.parquet
COPY data/spark_metadata/warehouse/click_events/events-002.parquet /data/spark_metadata/warehouse/click_events/events-002.parquet
COPY data/spark_metadata/warehouse/user_profiles/profiles.parquet /data/spark_metadata/warehouse/user_profiles/profiles.parquet
COPY data/spark_metadata/warehouse/payment_records/payments.parquet /data/spark_metadata/warehouse/payment_records/payments.parquet

WORKDIR /data/spark_metadata