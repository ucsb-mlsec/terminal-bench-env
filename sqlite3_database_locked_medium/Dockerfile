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
    apt-get install -y sqlite3 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/data /workspace/scripts

COPY data/metrics.db /workspace/data/metrics.db
COPY scripts/write_metrics.py /workspace/scripts/write_metrics.py
COPY scripts/read_metrics.py /workspace/scripts/read_metrics.py
COPY test_concurrent_access.sh /workspace/test_concurrent_access.sh

RUN chmod +x /workspace/test_concurrent_access.sh

WORKDIR /workspace