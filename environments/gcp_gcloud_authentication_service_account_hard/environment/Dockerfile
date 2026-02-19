FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data /data/subdir /data/backup

COPY data/valid-key-1.json /data/valid-key-1.json
COPY data/valid-key-2.json /data/valid-key-2.json
COPY data/wrong-type.json /data/wrong-type.json
COPY data/empty-field-values.json /data/empty-field-values.json
COPY data/subdir/malformed-key.json /data/subdir/malformed-key.json
COPY data/backup/old-valid-key.json /data/backup/old-valid-key.json
COPY data/random-file.txt /data/random-file.txt
COPY data/array-format.json /data/array-format.json

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
