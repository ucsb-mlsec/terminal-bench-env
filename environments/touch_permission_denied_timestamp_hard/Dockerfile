FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /data/archive/docs \
    /data/archive/logs \
    /data/archive/protected \
    /data/archive/backup \
    /data/archive/temp \
    /data/archive/readonly_mount \
    /solution

# Copy reference timestamp file
COPY data/reference.timestamp /data/reference.timestamp

# Copy files list
COPY data/files_to_sync.txt /data/files_to_sync.txt

# Copy archive files
COPY data/archive/docs/report1.txt /data/archive/docs/report1.txt
COPY data/archive/docs/report2.log /data/archive/docs/report2.log
COPY data/archive/logs/access.log /data/archive/logs/access.log
COPY data/archive/logs/error.log /data/archive/logs/error.log
COPY data/archive/protected/secret1.dat /data/archive/protected/secret1.dat
COPY data/archive/protected/secret2.dat /data/archive/protected/secret2.dat
COPY data/archive/protected/secret3.conf /data/archive/protected/secret3.conf
COPY data/archive/backup/file1.bak /data/archive/backup/file1.bak
COPY data/archive/backup/file2.bak /data/archive/backup/file2.bak
COPY data/archive/temp/data1.tmp /data/archive/temp/data1.tmp
COPY data/archive/temp/data2.tmp /data/archive/temp/data2.tmp
COPY data/archive/readonly_mount/locked1.txt /data/archive/readonly_mount/locked1.txt
COPY data/archive/readonly_mount/locked2.txt /data/archive/readonly_mount/locked2.txt
COPY data/archive/readonly_mount/locked3.log /data/archive/readonly_mount/locked3.log

# Mount the readonly_mount directory as read-only
RUN mount --bind /data/archive/readonly_mount /data/archive/readonly_mount || true && \
    mount -o remount,ro /data/archive/readonly_mount || true

WORKDIR /data