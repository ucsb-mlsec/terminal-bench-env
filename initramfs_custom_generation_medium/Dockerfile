FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/initramfs-template/modules \
    && mkdir -p /opt/initramfs-template/configs \
    && mkdir -p /tmp

COPY opt/initramfs-template/init-skeleton /opt/initramfs-template/init-skeleton
COPY opt/initramfs-template/modules/available-modules.txt /opt/initramfs-template/modules/available-modules.txt
COPY opt/initramfs-template/configs/server-example.conf /opt/initramfs-template/configs/server-example.conf
COPY opt/initramfs-template/configs/embedded-example.conf /opt/initramfs-template/configs/embedded-example.conf
COPY opt/initramfs-template/README /opt/initramfs-template/README

WORKDIR /tmp