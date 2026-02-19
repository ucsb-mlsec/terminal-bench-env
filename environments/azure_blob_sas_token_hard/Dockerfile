FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/downloads/batch1 \
    /workspace/downloads/batch2 \
    /workspace/downloads/batch3 \
    /workspace/downloads/partial \
    /workspace/downloads/archive \
    /workspace/downloads/recent \
    /workspace/downloads/mixed \
    /workspace/config \
    /workspace/output

COPY data/downloads/batch1/transactions_001.json /workspace/downloads/batch1/transactions_001.json
COPY data/downloads/batch1/transactions_002.json /workspace/downloads/batch1/transactions_002.json
COPY data/downloads/batch2/data_export.json /workspace/downloads/batch2/data_export.json
COPY data/downloads/batch2/corrupt_file.json /workspace/downloads/batch2/corrupt_file.json
COPY data/downloads/batch3/transactions_duplicate.json /workspace/downloads/batch3/transactions_duplicate.json
COPY data/downloads/partial/incomplete_download.json /workspace/downloads/partial/incomplete_download.json
COPY data/downloads/archive/old_transactions.json /workspace/downloads/archive/old_transactions.json
COPY data/downloads/recent/latest_batch.json /workspace/downloads/recent/latest_batch.json
COPY data/downloads/mixed/bad_customer_ids.json /workspace/downloads/mixed/bad_customer_ids.json
COPY data/downloads/mixed/invalid_timestamps.json /workspace/downloads/mixed/invalid_timestamps.json
COPY data/downloads/empty_file.json /workspace/downloads/empty_file.json
COPY config/container_manifest.json /workspace/config/container_manifest.json
COPY config/expired_credentials.yaml /workspace/config/expired_credentials.yaml
COPY config/download_log.txt /workspace/config/download_log.txt

WORKDIR /workspace