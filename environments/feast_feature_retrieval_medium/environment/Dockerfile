FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/feature_store/data /opt/feature_store/config

COPY data/customer_batch.txt /opt/feature_store/customer_batch.txt
COPY data/2024-01-15.json /opt/feature_store/data/2024-01-15.json
COPY data/2024-01-14.json /opt/feature_store/data/2024-01-14.json
COPY data/2024-01-16.json /opt/feature_store/data/2024-01-16.json
COPY config/feature_config.json /opt/feature_store/config/feature_config.json
COPY retrieve_features.py /opt/feature_store/retrieve_features.py

WORKDIR /opt/feature_store

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
