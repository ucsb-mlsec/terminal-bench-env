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

RUN pip3 install --no-cache-dir --break-system-packages feast pyarrow pandas

RUN mkdir -p /workspace/feast_setup/data /workspace/feast_setup/config

COPY data/users.parquet /workspace/feast_setup/data/users.parquet
COPY config/feature_store.yaml /workspace/feast_setup/feature_store.yaml
COPY features.py /workspace/feast_setup/features.py

WORKDIR /workspace