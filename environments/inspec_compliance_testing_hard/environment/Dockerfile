FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /compliance /system/config /workspace

COPY data/compliance/report.json /compliance/report.json
COPY data/system/config/ssh_config.conf /system/config/ssh_config.conf
COPY data/system/config/users.json /system/config/users.json
COPY data/system/config/firewall.rules /system/config/firewall.rules
COPY data/system/config/services.conf /system/config/services.conf
COPY data/system/config/file_permissions.txt /system/config/file_permissions.txt
COPY data/system/config/web_server.conf /system/config/web_server.conf
COPY data/system/config/audit.conf /system/config/audit.conf

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
