FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/user/data_processor/data

COPY process_transactions.py /home/user/data_processor/process_transactions.py
COPY data/transactions_small.csv /home/user/data_processor/transactions_small.csv
COPY data/transactions_medium.csv /home/user/data_processor/transactions_medium.csv
COPY data/transactions_large.csv /home/user/data_processor/transactions_large.csv
COPY benchmark.py /home/user/data_processor/benchmark.py

WORKDIR /home/user/data_processor