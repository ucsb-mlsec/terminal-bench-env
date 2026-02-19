FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/monitoring/scripts /opt/monitoring/data

COPY opt/monitoring/scripts/cpu_monitor.sh /opt/monitoring/scripts/cpu_monitor.sh
COPY opt/monitoring/scripts/memory_check.sh /opt/monitoring/scripts/memory_check.sh
COPY opt/monitoring/scripts/disk_space.sh /opt/monitoring/scripts/disk_space.sh
COPY opt/monitoring/scripts/process_count.sh /opt/monitoring/scripts/process_count.sh
COPY opt/monitoring/scripts/load_average.sh /opt/monitoring/scripts/load_average.sh

RUN chmod +x /opt/monitoring/scripts/*.sh

WORKDIR /opt/monitoring