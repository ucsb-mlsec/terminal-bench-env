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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages dask[complete] pandas

RUN mkdir -p /workspace/data

COPY process_data.py /workspace/process_data.py
COPY data/transactions_2023_01.csv /workspace/data/transactions_2023_01.csv
COPY data/transactions_2023_02.csv /workspace/data/transactions_2023_02.csv
COPY data/transactions_2023_03.csv /workspace/data/transactions_2023_03.csv

WORKDIR /workspace