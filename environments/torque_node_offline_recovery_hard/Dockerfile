FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/log/torque /root

COPY data/server.log /var/log/torque/server.log
COPY data/node01.log /var/log/torque/node01.log
COPY data/node02.log /var/log/torque/node02.log
COPY data/node03.log /var/log/torque/node03.log
COPY data/node04.log /var/log/torque/node04.log
COPY data/node05.log /var/log/torque/node05.log
COPY data/node06.log /var/log/torque/node06.log
COPY data/node07.log /var/log/torque/node07.log
COPY data/node08.log /var/log/torque/node08.log
COPY data/node09.log /var/log/torque/node09.log
COPY data/node10.log /var/log/torque/node10.log
COPY data/node11.log /var/log/torque/node11.log
COPY data/node12.log /var/log/torque/node12.log

WORKDIR /root