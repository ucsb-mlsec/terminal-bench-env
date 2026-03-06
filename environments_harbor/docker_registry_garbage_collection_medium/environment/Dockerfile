FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/storage/blobs /solution

COPY data/manifest.txt /var/storage/manifest.txt
COPY data/blob_a1b2c3.dat /var/storage/blobs/blob_a1b2c3.dat
COPY data/blob_4f5e6d.dat /var/storage/blobs/blob_4f5e6d.dat
COPY data/blob_xyz123.dat /var/storage/blobs/blob_xyz123.dat
COPY data/data_m9n8p7.bin /var/storage/blobs/data_m9n8p7.bin
COPY data/blob_old001.dat /var/storage/blobs/blob_old001.dat
COPY data/blob_old002.dat /var/storage/blobs/blob_old002.dat
COPY data/data_legacy99.bin /var/storage/blobs/data_legacy99.bin
COPY data/blob_unused55.dat /var/storage/blobs/blob_unused55.dat
COPY data/data_orphan.bin /var/storage/blobs/data_orphan.bin

WORKDIR /var/storage

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
