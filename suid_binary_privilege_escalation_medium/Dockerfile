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
RUN mkdir -p /root /usr/local/bin /tmp

# Copy the secret file to root directory
COPY root/secret.txt /root/secret.txt

# Copy the maintenance script to the expected location
COPY usr/local/bin/maintenance.sh /usr/local/bin/maintenance.sh

# Set proper permissions for the secret file (only root can read)
RUN chmod 600 /root/secret.txt && \
    chown root:root /root/secret.txt

# Set SUID permissions on the maintenance script and make it executable
RUN chmod 4755 /usr/local/bin/maintenance.sh && \
    chown root:root /usr/local/bin/maintenance.sh

# Set working directory
WORKDIR /workspace