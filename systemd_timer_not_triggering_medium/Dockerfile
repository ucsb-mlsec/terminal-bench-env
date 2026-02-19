FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install cron
RUN apt-get update && \
    apt-get install -y cron && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gzip findutils && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /etc/cron.d && \
    mkdir -p /usr/local/bin && \
    mkdir -p /var/log/myapp/archive && \
    mkdir -p /root

# Copy the cron configuration file
COPY config/log-rotation /etc/cron.d/log-rotation

# Copy the rotation script
COPY rotate-logs.sh /usr/local/bin/rotate-logs.sh

# Copy application logs
COPY data/app.log /var/log/myapp/app.log
COPY data/app-2024-01-14.log /var/log/myapp/app-2024-01-14.log

# Set proper permissions
RUN chmod 0644 /etc/cron.d/log-rotation && \
    chmod +x /usr/local/bin/rotate-logs.sh && \
    chmod 755 /var/log/myapp && \
    chmod 755 /var/log/myapp/archive

WORKDIR /root