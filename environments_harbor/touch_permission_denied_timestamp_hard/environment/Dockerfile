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
RUN mkdir -p /workspace/data/archive/docs \
    /workspace/data/archive/logs \
    /workspace/data/archive/protected \
    /workspace/data/archive/backup \
    /workspace/data/archive/temp \
    /workspace/data/archive/readonly_mount \
    /workspace/solution

# Copy reference timestamp file
COPY data/reference.timestamp /workspace/data/reference.timestamp

# Copy files list
COPY data/files_to_sync.txt /workspace/data/files_to_sync.txt

# Copy archive files
COPY data/archive/docs/report1.txt /workspace/data/archive/docs/report1.txt
COPY data/archive/docs/report2.log /workspace/data/archive/docs/report2.log
COPY data/archive/logs/access.log /workspace/data/archive/logs/access.log
COPY data/archive/logs/error.log /workspace/data/archive/logs/error.log
COPY data/archive/protected/secret1.dat /workspace/data/archive/protected/secret1.dat
COPY data/archive/protected/secret2.dat /workspace/data/archive/protected/secret2.dat
COPY data/archive/protected/secret3.conf /workspace/data/archive/protected/secret3.conf
COPY data/archive/backup/file1.bak /workspace/data/archive/backup/file1.bak
COPY data/archive/backup/file2.bak /workspace/data/archive/backup/file2.bak
COPY data/archive/temp/data1.tmp /workspace/data/archive/temp/data1.tmp
COPY data/archive/temp/data2.tmp /workspace/data/archive/temp/data2.tmp
COPY data/archive/readonly_mount/locked1.txt /workspace/data/archive/readonly_mount/locked1.txt
COPY data/archive/readonly_mount/locked2.txt /workspace/data/archive/readonly_mount/locked2.txt
COPY data/archive/readonly_mount/locked3.log /workspace/data/archive/readonly_mount/locked3.log

# Mount the readonly_mount directory as read-only
RUN mount --bind /workspace/data/archive/readonly_mount /workspace/data/archive/readonly_mount || true && \
    mount -o remount,ro /workspace/data/archive/readonly_mount || true

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
