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
RUN pip3 install --no-cache-dir --break-system-packages great-expectations>=0.15.0 pyarrow

RUN pip3 install --no-cache-dir --break-system-packages \
    great-expectations \
    pandas \
    pyarrow \
    openpyxl

RUN mkdir -p /data/sources /solution

COPY data/sources/transactions_batch1.csv /data/sources/transactions_batch1.csv
COPY data/sources/transactions_batch2.csv /data/sources/transactions_batch2.csv
COPY data/sources/api_transactions.json /data/sources/api_transactions.json
COPY data/sources/db_export.parquet /data/sources/db_export.parquet

WORKDIR /solution