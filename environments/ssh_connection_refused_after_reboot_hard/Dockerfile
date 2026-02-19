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
    apt-get install -y openssh-server systemd && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/systemd/system/ssh.service.d && \
    mkdir -p /lib/systemd/system && \
    mkdir -p /root

COPY etc/systemd/system/ssh.service.d/override.conf /etc/systemd/system/ssh.service.d/override.conf
COPY lib/systemd/system/ssh.service /lib/systemd/system/ssh.service

WORKDIR /root