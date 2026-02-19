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
RUN mkdir -p /opt/monitor/collectors /solution

# Copy monitoring scripts to their expected locations
COPY opt/monitor/run_monitors.sh /opt/monitor/run_monitors.sh
COPY opt/monitor/collectors/network_monitor.sh /opt/monitor/collectors/network_monitor.sh
COPY opt/monitor/collectors/disk_monitor.sh /opt/monitor/collectors/disk_monitor.sh
COPY opt/monitor/collectors/log_monitor.sh /opt/monitor/collectors/log_monitor.sh

# Make all scripts executable
RUN chmod +x /opt/monitor/run_monitors.sh \
    /opt/monitor/collectors/network_monitor.sh \
    /opt/monitor/collectors/disk_monitor.sh \
    /opt/monitor/collectors/log_monitor.sh

WORKDIR /solution