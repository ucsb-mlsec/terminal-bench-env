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
    apt-get install -y netcat-openbsd jq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data /config /home/user

COPY data/security_incident.log /data/security_incident.log
COPY data/network_snapshot.txt /data/network_snapshot.txt
COPY data/process_dump.json /data/process_dump.json
COPY config/firewall_rules.conf /config/firewall_rules.conf
COPY data/connection_attempts.csv /data/connection_attempts.csv
COPY data/corrupted_timeline.txt /data/corrupted_timeline.txt

WORKDIR /home/user