FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/crash/monitoring-incident /home/agent /workspace

COPY data/process_snapshot.txt /var/crash/monitoring-incident/process_snapshot.txt
COPY data/job_table.txt /var/crash/monitoring-incident/job_table.txt
COPY config/critical_services.conf /var/crash/monitoring-incident/critical_services.conf

WORKDIR /workspace