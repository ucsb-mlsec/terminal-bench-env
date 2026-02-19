FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/libvirt_snapshot /root

COPY data/pools.xml /opt/libvirt_snapshot/pools.xml
COPY data/permissions.txt /opt/libvirt_snapshot/permissions.txt
COPY data/qemu_config.txt /opt/libvirt_snapshot/qemu_config.txt
COPY data/vm_errors.log /opt/libvirt_snapshot/vm_errors.log
COPY data/apparmor_status.txt /opt/libvirt_snapshot/apparmor_status.txt

WORKDIR /root