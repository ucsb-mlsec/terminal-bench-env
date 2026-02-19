FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y tar gzip procps && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directories
RUN mkdir -p /var/lock \
    && mkdir -p /srv/application/data \
    && mkdir -p /backup/archive \
    && mkdir -p /usr/local/bin \
    && mkdir -p /root

# Copy generated files to their expected locations
COPY data/file1.txt /srv/application/data/file1.txt
COPY data/file2.txt /srv/application/data/file2.txt
COPY data/file3.txt /srv/application/data/file3.txt
COPY backup.sh /usr/local/bin/backup.sh

# Make backup script executable
RUN chmod +x /usr/local/bin/backup.sh

# Create a stale lock file with a non-existent PID
RUN echo "99999" > /var/lock/backup.lock

WORKDIR /root