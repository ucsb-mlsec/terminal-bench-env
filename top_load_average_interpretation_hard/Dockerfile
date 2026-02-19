FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/diagnostics

COPY data/diagnostics/top_snapshot.txt /home/agent/diagnostics/top_snapshot.txt
COPY data/diagnostics/vmstat_snapshot.txt /home/agent/diagnostics/vmstat_snapshot.txt
COPY data/diagnostics/iostat_snapshot.txt /home/agent/diagnostics/iostat_snapshot.txt
COPY data/diagnostics/ps_snapshot.txt /home/agent/diagnostics/ps_snapshot.txt

WORKDIR /home/agent