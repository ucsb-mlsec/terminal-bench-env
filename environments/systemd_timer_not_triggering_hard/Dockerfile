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
    apt-get install -y systemd && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/systemd/system /usr/local/bin /root

COPY etc/systemd/system/database-backup.timer /etc/systemd/system/database-backup.timer
COPY etc/systemd/system/database-backup.service /etc/systemd/system/database-backup.service
COPY usr/local/bin/backup-database.sh /usr/local/bin/backup-database.sh

RUN chmod +x /usr/local/bin/backup-database.sh

WORKDIR /root