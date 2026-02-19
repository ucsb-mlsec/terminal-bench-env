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
    apt-get install -y --no-install-recommends \
    grub2-common \
    grub-common && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /boot/grub /boot /etc/default /root

COPY boot/grub/grub.cfg /boot/grub/grub.cfg
COPY etc/default/grub /etc/default/grub
COPY boot/vmlinuz-5.15.0-91-generic /boot/vmlinuz-5.15.0-91-generic
COPY boot/vmlinuz-5.15.0-92-generic /boot/vmlinuz-5.15.0-92-generic
COPY boot/vmlinuz-5.15.0-93-generic /boot/vmlinuz-5.15.0-93-generic
COPY boot/initrd.img-5.15.0-91-generic /boot/initrd.img-5.15.0-91-generic
COPY boot/initrd.img-5.15.0-92-generic /boot/initrd.img-5.15.0-92-generic
COPY boot/initrd.img-5.15.0-93-generic /boot/initrd.img-5.15.0-93-generic

WORKDIR /root