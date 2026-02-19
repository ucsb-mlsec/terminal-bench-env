FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/logs/application/2024-01 \
    /var/logs/application/2024-02 \
    /var/logs/application/2024-03 \
    /home/agent

COPY data/var/logs/application/2024-01/app-2024-01-15.log /var/logs/application/2024-01/app-2024-01-15.log
COPY data/var/logs/application/2024-01/app-2024-01-16.log /var/logs/application/2024-01/app-2024-01-16.log
COPY data/var/logs/application/2024-01/app-2024-01-17.log /var/logs/application/2024-01/app-2024-01-17.log
COPY data/var/logs/application/2024-02/app-2024-02-01.log /var/logs/application/2024-02/app-2024-02-01.log
COPY data/var/logs/application/2024-02/app-2024-02-02.log /var/logs/application/2024-02/app-2024-02-02.log
COPY data/var/logs/application/2024-02/app-2024-02-03.log /var/logs/application/2024-02/app-2024-02-03.log
COPY data/var/logs/application/2024-03/app-2024-03-01.log /var/logs/application/2024-03/app-2024-03-01.log
COPY data/var/logs/application/2024-03/app-2024-03-02.log /var/logs/application/2024-03/app-2024-03-02.log
COPY data/var/logs/application/2024-03/info-2024-03-03.txt /var/logs/application/2024-03/info-2024-03-03.txt
COPY data/var/logs/application/2024-03/app-2024-03-04.log /var/logs/application/2024-03/app-2024-03-04.log

WORKDIR /home/agent