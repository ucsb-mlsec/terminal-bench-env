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

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages delta-spark pyspark

RUN pip3 install --no-cache-dir --break-system-packages \
    delta-spark \
    pyspark \
    pandas \
    pyarrow

RUN mkdir -p /workspace/catalog_delta/_delta_log

COPY data/catalog_delta/_delta_log/00000000000000000000.json /workspace/catalog_delta/_delta_log/00000000000000000000.json
COPY data/catalog_delta/_delta_log/00000000000000000001.json /workspace/catalog_delta/_delta_log/00000000000000000001.json
COPY data/catalog_delta/_delta_log/00000000000000000002.json /workspace/catalog_delta/_delta_log/00000000000000000002.json
COPY data/catalog_delta/_delta_log/00000000000000000003.json /workspace/catalog_delta/_delta_log/00000000000000000003.json
COPY data/catalog_delta/_delta_log/00000000000000000004.json /workspace/catalog_delta/_delta_log/00000000000000000004.json
COPY data/catalog_delta/part-00000-a1b2c3d4.snappy.parquet /workspace/catalog_delta/part-00000-a1b2c3d4.snappy.parquet
COPY data/catalog_delta/part-00001-b2c3d4e5.snappy.parquet /workspace/catalog_delta/part-00001-b2c3d4e5.snappy.parquet
COPY data/catalog_delta/part-00002-c3d4e5f6.snappy.parquet /workspace/catalog_delta/part-00002-c3d4e5f6.snappy.parquet
COPY data/catalog_delta/part-00003-d4e5f6g7.snappy.parquet /workspace/catalog_delta/part-00003-d4e5f6g7.snappy.parquet
COPY data/catalog_delta/part-00004-e5f6g7h8.snappy.parquet /workspace/catalog_delta/part-00004-e5f6g7h8.snappy.parquet

WORKDIR /workspace