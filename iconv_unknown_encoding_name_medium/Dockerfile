FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y libc-bin && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/legacy_data

COPY data/legacy_data/customer_records.txt /workspace/legacy_data/customer_records.txt
COPY data/legacy_data/product_catalog.txt /workspace/legacy_data/product_catalog.txt
COPY data/legacy_data/transaction_log.txt /workspace/legacy_data/transaction_log.txt

WORKDIR /workspace