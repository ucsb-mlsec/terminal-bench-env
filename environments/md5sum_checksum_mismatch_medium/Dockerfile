FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /backup/configs /solution

COPY data/checksums.md5 /backup/configs/checksums.md5
COPY data/database.conf /backup/configs/database.conf
COPY data/network.cfg /backup/configs/network.cfg
COPY data/app.config /backup/configs/app.config
COPY data/system.ini /backup/configs/system.ini
COPY data/security.yaml /backup/configs/security.yaml
COPY data/backup.json /backup/configs/backup.json
COPY data/server.properties /backup/configs/server.properties
COPY data/credentials.env /backup/configs/credentials.env
COPY data/settings.xml /backup/configs/settings.xml
COPY data/firewall.rules /backup/configs/firewall.rules

WORKDIR /backup/configs