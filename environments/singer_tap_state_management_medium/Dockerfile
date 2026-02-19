FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/singer/extraction_logs

COPY data/singer/state_backup.json /data/singer/state_backup.json
COPY data/singer/extraction_logs/tap_output.log /data/singer/extraction_logs/tap_output.log
COPY data/singer/extraction_logs/state_updates.log /data/singer/extraction_logs/state_updates.log
COPY data/singer/catalog.json /data/singer/catalog.json

WORKDIR /data/singer