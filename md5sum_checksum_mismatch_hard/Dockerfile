FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /backup/archives/2024-01-15 \
    /backup/archives/2024-01-16 \
    /backup/archives/2024-01-17 \
    /backup/archives/2024-02-10 \
    /backup/archives/2024-02-11 \
    /backup/archives/2024-02-12 \
    /backup/archives/2024-03-20 \
    /backup/archives/2024-03-21 \
    /backup/archives/2024-03-22 \
    /backup/checksums/2024-01-15 \
    /backup/checksums/2024-01-16 \
    /backup/checksums/2024-01-17 \
    /backup/checksums/2024-02-10 \
    /backup/checksums/2024-02-11 \
    /backup/checksums/2024-02-12 \
    /backup/checksums/2024-03-20 \
    /backup/checksums/2024-03-21 \
    /backup/checksums/2024-03-22 \
    /root

COPY data/backup_archives_2024-01-15.tar.gz /backup/archives/2024-01-15/
COPY data/backup_archives_2024-01-16.tar.gz /backup/archives/2024-01-16/
COPY data/backup_archives_2024-01-17.tar.gz /backup/archives/2024-01-17/
COPY data/backup_archives_2024-02-10.tar.gz /backup/archives/2024-02-10/
COPY data/backup_archives_2024-02-11.tar.gz /backup/archives/2024-02-11/
COPY data/backup_archives_2024-02-12.tar.gz /backup/archives/2024-02-12/
COPY data/backup_archives_2024-03-20.tar.gz /backup/archives/2024-03-20/
COPY data/backup_archives_2024-03-21.tar.gz /backup/archives/2024-03-21/
COPY data/backup_archives_2024-03-22.tar.gz /backup/archives/2024-03-22/

COPY data/backup_checksums_2024-01-15.md5 /backup/checksums/2024-01-15/
COPY data/backup_checksums_2024-01-16.md5 /backup/checksums/2024-01-16/
COPY data/backup_checksums_2024-01-17.md5 /backup/checksums/2024-01-17/
COPY data/backup_checksums_2024-02-10.md5 /backup/checksums/2024-02-10/
COPY data/backup_checksums_2024-02-11.md5 /backup/checksums/2024-02-11/
COPY data/backup_checksums_2024-02-12.md5 /backup/checksums/2024-02-12/
COPY data/backup_checksums_2024-03-20.md5 /backup/checksums/2024-03-20/
COPY data/backup_checksums_2024-03-21.md5 /backup/checksums/2024-03-21/
COPY data/backup_checksums_2024-03-22.md5 /backup/checksums/2024-03-22/

WORKDIR /root