FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data/transactions/versions/v1 \
    /workspace/data/transactions/versions/v2 \
    /workspace/data/transactions/versions/v3 \
    /workspace/data/transactions/versions/v4 \
    /workspace/data/transactions/versions/v5 \
    /workspace/data/transactions/versions/v6 \
    /workspace/data/transactions/versions/v7 \
    /workspace/data/transactions/versions/v8 \
    /workspace/solution

COPY data/transactions/versions/v1/transactions.csv /workspace/data/transactions/versions/v1/transactions.csv
COPY data/transactions/versions/v2/transactions.csv /workspace/data/transactions/versions/v2/transactions.csv
COPY data/transactions/versions/v3/transactions.csv /workspace/data/transactions/versions/v3/transactions.csv
COPY data/transactions/versions/v4/transactions.csv /workspace/data/transactions/versions/v4/transactions.csv
COPY data/transactions/versions/v5/transactions.csv /workspace/data/transactions/versions/v5/transactions.csv
COPY data/transactions/versions/v6/transactions.csv /workspace/data/transactions/versions/v6/transactions.csv
COPY data/transactions/versions/v7/transactions.csv /workspace/data/transactions/versions/v7/transactions.csv
COPY data/transactions/versions/v8/transactions.csv /workspace/data/transactions/versions/v8/transactions.csv

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
