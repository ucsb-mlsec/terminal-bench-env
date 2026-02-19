FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y cron && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/spool/cron/crontabs \
    /opt/webapp \
    /home/webapp \
    /var/lib/webapp/data \
    /var/backups/webapp \
    /usr/local/bin

COPY var/spool/cron/crontabs/webapp /var/spool/cron/crontabs/webapp
COPY opt/webapp/backup.sh /opt/webapp/backup.sh
COPY home/webapp/.profile /home/webapp/.profile

RUN chmod +x /opt/webapp/backup.sh \
    && chmod 600 /var/spool/cron/crontabs/webapp

WORKDIR /home/webapp