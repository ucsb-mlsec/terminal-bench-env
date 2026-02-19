FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/deadlock_investigation

COPY data/schema.sql /home/agent/deadlock_investigation/schema.sql
COPY data/deadlock_log.txt /home/agent/deadlock_investigation/deadlock_log.txt
COPY data/transaction_sequences.sql /home/agent/deadlock_investigation/transaction_sequences.sql

WORKDIR /home/agent