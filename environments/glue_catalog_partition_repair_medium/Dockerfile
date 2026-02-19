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

RUN pip3 install --no-cache-dir --break-system-packages pandas pyarrow

RUN mkdir -p /data/glue-catalog /data/s3-bucket/sales_transactions /solution

COPY data/glue-catalog/sales_transactions_catalog.json /data/glue-catalog/sales_transactions_catalog.json
COPY data/s3-bucket/sales_transactions/year=2024/month=01/day=15/data_001.parquet /data/s3-bucket/sales_transactions/year=2024/month=01/day=15/data_001.parquet
COPY data/s3-bucket/sales_transactions/year=2024/month=01/day=20/data_001.parquet /data/s3-bucket/sales_transactions/year=2024/month=01/day=20/data_001.parquet
COPY data/s3-bucket/sales_transactions/year=2024/month=01/day=22/data_001.parquet /data/s3-bucket/sales_transactions/year=2024/month=01/day=22/data_001.parquet
COPY data/s3-bucket/sales_transactions/year=2024/month=02/day=05/data_001.parquet /data/s3-bucket/sales_transactions/year=2024/month=02/day=05/data_001.parquet
COPY data/s3-bucket/sales_transactions/year=2024/month=02/day=10/data_001.parquet /data/s3-bucket/sales_transactions/year=2024/month=02/day=10/data_001.parquet
COPY data/s3-bucket/sales_transactions/year=2024/month=02/day=28/data_001.parquet /data/s3-bucket/sales_transactions/year=2024/month=02/day=28/data_001.parquet
COPY data/s3-bucket/sales_transactions/year=2024/month=03/day=05/data_001.parquet /data/s3-bucket/sales_transactions/year=2024/month=03/day=05/data_001.parquet
COPY data/s3-bucket/sales_transactions/year=2024/month=03/day=12/data_001.parquet /data/s3-bucket/sales_transactions/year=2024/month=03/day=12/data_001.parquet
COPY data/s3-bucket/sales_transactions/year=2024/month=03/day=25/data_001.parquet /data/s3-bucket/sales_transactions/year=2024/month=03/day=25/data_001.parquet

WORKDIR /workspace