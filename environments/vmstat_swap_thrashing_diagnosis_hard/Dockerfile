FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pandas numpy

RUN mkdir -p /opt/monitoring/system_state /home/agent

COPY data/vmstat_24h.log /opt/monitoring/vmstat_24h.log
COPY data/system_state/meminfo.snapshot /opt/monitoring/system_state/meminfo.snapshot
COPY data/system_state/top_processes.snapshot /opt/monitoring/system_state/top_processes.snapshot
COPY data/system_state/sysctl_memory.conf /opt/monitoring/system_state/sysctl_memory.conf
COPY data/system_state/database_config.ini /opt/monitoring/system_state/database_config.ini

WORKDIR /home/agent