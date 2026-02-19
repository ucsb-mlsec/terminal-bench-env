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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages openpyxl

RUN mkdir -p /workspace/raw_data /workspace/templates /workspace/output /workspace/solution

COPY data/transactions_001.json /workspace/raw_data/transactions_001.json
COPY data/transactions_002.json /workspace/raw_data/transactions_002.json
COPY data/transactions_003.json /workspace/raw_data/transactions_003.json

WORKDIR /workspace