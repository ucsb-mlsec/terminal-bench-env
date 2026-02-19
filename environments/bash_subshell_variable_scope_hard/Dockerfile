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
    apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/monitor /var/logs/app

COPY process_logs.sh /opt/monitor/process_logs.sh
COPY data/server1.log /var/logs/app/server1.log
COPY data/server2.log /var/logs/app/server2.log
COPY data/server3.log /var/logs/app/server3.log

RUN chmod +x /opt/monitor/process_logs.sh

WORKDIR /opt/monitor