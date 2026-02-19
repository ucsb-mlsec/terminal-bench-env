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
    apt-get install -y mdadm && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /raid_diagnostics /root

COPY data/raid_diagnostics/device_sda.txt /raid_diagnostics/device_sda.txt
COPY data/raid_diagnostics/device_sdb.txt /raid_diagnostics/device_sdb.txt
COPY data/raid_diagnostics/device_sdc.txt /raid_diagnostics/device_sdc.txt
COPY data/raid_diagnostics/device_sdd.txt /raid_diagnostics/device_sdd.txt

WORKDIR /root