FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/data

COPY batch_processor.py /workspace/batch_processor.py
COPY data/batch_001.csv /workspace/data/batch_001.csv
COPY data/batch_002.csv /workspace/data/batch_002.csv
COPY data/batch_003.csv /workspace/data/batch_003.csv
COPY data/batch_004.csv /workspace/data/batch_004.csv
COPY data/batch_005.csv /workspace/data/batch_005.csv

WORKDIR /workspace