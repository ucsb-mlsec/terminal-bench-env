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

RUN pip3 install --no-cache-dir --break-system-packages ray[default] pandas pyarrow

RUN mkdir -p /workspace/data/transactions/2023 /workspace/data/transactions/2024 /workspace/output

COPY data/transactions/2023/txn_2023_01.parquet /workspace/data/transactions/2023/txn_2023_01.parquet
COPY data/transactions/2023/txn_2023_02.parquet /workspace/data/transactions/2023/txn_2023_02.parquet
COPY data/transactions/2023/txn_2023_03.parquet /workspace/data/transactions/2023/txn_2023_03.parquet
COPY data/transactions/2023/txn_2023_04.parquet /workspace/data/transactions/2023/txn_2023_04.parquet
COPY data/transactions/2023/txn_2023_05.parquet /workspace/data/transactions/2023/txn_2023_05.parquet
COPY data/transactions/2023/txn_2023_06.parquet /workspace/data/transactions/2023/txn_2023_06.parquet
COPY data/transactions/2023/txn_2023_07.parquet /workspace/data/transactions/2023/txn_2023_07.parquet
COPY data/transactions/2023/txn_2023_08.parquet /workspace/data/transactions/2023/txn_2023_08.parquet
COPY data/transactions/2023/txn_2023_09.parquet /workspace/data/transactions/2023/txn_2023_09.parquet
COPY data/transactions/2023/txn_2023_10.parquet /workspace/data/transactions/2023/txn_2023_10.parquet
COPY data/transactions/2024/txn_2024_01.parquet /workspace/data/transactions/2024/txn_2024_01.parquet
COPY data/transactions/2024/txn_2024_02.parquet /workspace/data/transactions/2024/txn_2024_02.parquet
COPY data/transactions/2024/txn_2024_03.parquet /workspace/data/transactions/2024/txn_2024_03.parquet
COPY data/transactions/2024/txn_2024_04.parquet /workspace/data/transactions/2024/txn_2024_04.parquet
COPY data/transactions/2024/txn_2024_05.parquet /workspace/data/transactions/2024/txn_2024_05.parquet
COPY data/transactions/2024/txn_2024_06.parquet /workspace/data/transactions/2024/txn_2024_06.parquet
COPY data/transactions/2024/txn_2024_07.parquet /workspace/data/transactions/2024/txn_2024_07.parquet
COPY data/transactions/2024/txn_2024_08.parquet /workspace/data/transactions/2024/txn_2024_08.parquet
COPY data/transactions/2024/txn_2024_09.parquet /workspace/data/transactions/2024/txn_2024_09.parquet
COPY data/transactions/2024/txn_2024_10.parquet /workspace/data/transactions/2024/txn_2024_10.parquet

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
