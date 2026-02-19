FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /storage/public-data \
    /storage/app-logs \
    /storage/private-docs \
    /storage/backups \
    /storage/media \
    /storage/temp-uploads \
    /storage/archives \
    /solution

COPY data/storage/public-data/.access /storage/public-data/.access
COPY data/storage/public-data/readme.txt /storage/public-data/readme.txt
COPY data/storage/public-data/config.txt /storage/public-data/config.txt
COPY data/storage/public-data/users.csv /storage/public-data/users.csv

COPY data/storage/app-logs/.access /storage/app-logs/.access
COPY data/storage/app-logs/application.log /storage/app-logs/application.log
COPY data/storage/app-logs/metrics.json /storage/app-logs/metrics.json

COPY data/storage/private-docs/.access /storage/private-docs/.access
COPY data/storage/private-docs/credentials.txt /storage/private-docs/credentials.txt

COPY data/storage/backups/.access /storage/backups/.access
COPY data/storage/backups/backup.sql /storage/backups/backup.sql
COPY data/storage/backups/env.properties /storage/backups/env.properties

COPY data/storage/media/.access /storage/media/.access
COPY data/storage/media/image001.txt /storage/media/image001.txt
COPY data/storage/media/video_metadata.json /storage/media/video_metadata.json

COPY data/storage/temp-uploads/.access /storage/temp-uploads/.access
COPY data/storage/temp-uploads/upload.txt /storage/temp-uploads/upload.txt
COPY data/storage/temp-uploads/settings.ini /storage/temp-uploads/settings.ini

COPY data/storage/archives/.access /storage/archives/.access
COPY data/storage/archives/old_config.yaml /storage/archives/old_config.yaml

WORKDIR /workspace