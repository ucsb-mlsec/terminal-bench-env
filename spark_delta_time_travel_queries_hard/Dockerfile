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

RUN pip3 install --no-cache-dir --break-system-packages pandas pyarrow

RUN mkdir -p /data/sales_versions /solution

COPY data/sales_versions/sales_v001_20240101.parquet /data/sales_versions/sales_v001_20240101.parquet
COPY data/sales_versions/sales_v002_20240102.parquet /data/sales_versions/sales_v002_20240102.parquet
COPY data/sales_versions/sales_v003_20240103.parquet /data/sales_versions/sales_v003_20240103.parquet
COPY data/sales_versions/sales_v004_20240104.parquet /data/sales_versions/sales_v004_20240104.parquet
COPY data/sales_versions/sales_v005_20240105.parquet /data/sales_versions/sales_v005_20240105.parquet
COPY data/sales_versions/sales_v006_20240106.parquet /data/sales_versions/sales_v006_20240106.parquet
COPY data/sales_versions/sales_v007_20240107.parquet /data/sales_versions/sales_v007_20240107.parquet

WORKDIR /workspace