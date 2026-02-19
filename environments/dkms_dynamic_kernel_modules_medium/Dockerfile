FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directory structure
RUN mkdir -p /usr/src \
    && mkdir -p /var/lib/dkms/nvidia/450.102/build \
    && mkdir -p /var/lib/dkms/vboxdrv/6.1.32/build \
    && mkdir -p /var/lib/dkms/rtl8821ce/5.5.8/build \
    && mkdir -p /var/lib/dkms/bbswitch/0.8/build \
    && mkdir -p /root

# Copy usr/src files
COPY usr/src/nvidia-390.48/dkms.conf /usr/src/nvidia-390.48/dkms.conf
COPY usr/src/nvidia-450.102/dkms.conf /usr/src/nvidia-450.102/dkms.conf
COPY usr/src/vboxdrv-6.0.14/dkms.conf /usr/src/vboxdrv-6.0.14/dkms.conf
COPY usr/src/vboxdrv-6.1.32/dkms.conf /usr/src/vboxdrv-6.1.32/dkms.conf
COPY usr/src/rtl8821ce-5.5.2/dkms.conf /usr/src/rtl8821ce-5.5.2/dkms.conf
COPY usr/src/rtl8821ce-5.5.8/dkms.conf /usr/src/rtl8821ce-5.5.8/dkms.conf
COPY usr/src/bbswitch-0.8/dkms.conf /usr/src/bbswitch-0.8/dkms.conf
COPY usr/src/linux-headers-5.4.0/Makefile /usr/src/linux-headers-5.4.0/Makefile

# Copy var/lib/dkms files
COPY var/lib/dkms/nvidia/450.102/build/module_built /var/lib/dkms/nvidia/450.102/build/module_built
COPY var/lib/dkms/vboxdrv/6.1.32/build/module_built /var/lib/dkms/vboxdrv/6.1.32/build/module_built
COPY var/lib/dkms/rtl8821ce/5.5.8/build/module_built /var/lib/dkms/rtl8821ce/5.5.8/build/module_built
COPY var/lib/dkms/bbswitch/0.8/build/module_built /var/lib/dkms/bbswitch/0.8/build/module_built

WORKDIR /root