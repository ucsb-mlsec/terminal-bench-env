FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y cron jq && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /var/lib/pool-registry \
    /var/log \
    /usr/local/bin \
    /root

# Copy generated files to their required locations
COPY data/pools.json /var/lib/pool-registry/pools.json
COPY pool-scrub /usr/local/bin/pool-scrub

# Make pool-scrub executable
RUN chmod +x /usr/local/bin/pool-scrub

# Ensure cron service can run
RUN touch /var/log/pool-scrubs.log && \
    chmod 644 /var/log/pool-scrubs.log

WORKDIR /root