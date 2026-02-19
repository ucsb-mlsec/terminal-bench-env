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
    apt-get install -y openssh-client && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/backup/ssh_configs /etc/ssh /home/admin

COPY data/verified_keys.txt /etc/ssh/verified_keys.txt
COPY data/known_hosts_1 /var/backup/ssh_configs/known_hosts_1
COPY data/known_hosts_2 /var/backup/ssh_configs/known_hosts_2
COPY data/known_hosts_3 /var/backup/ssh_configs/known_hosts_3
COPY data/known_hosts_4 /var/backup/ssh_configs/known_hosts_4
COPY data/known_hosts_5 /var/backup/ssh_configs/known_hosts_5

WORKDIR /home/admin

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
