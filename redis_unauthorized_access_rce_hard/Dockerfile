FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/evidence/suspicious_files

COPY data/redis_traffic.txt /home/evidence/redis_traffic.txt
COPY data/redis.conf /home/evidence/redis.conf
COPY data/system_logs.txt /home/evidence/system_logs.txt
COPY data/suspicious_files/cron_backup /home/evidence/suspicious_files/cron_backup
COPY data/suspicious_files/redis_dump.rdb /home/evidence/suspicious_files/redis_dump.rdb

WORKDIR /home/evidence