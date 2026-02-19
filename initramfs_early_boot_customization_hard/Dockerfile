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
    apt-get install -y \
    initramfs-tools \
    kmod \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /mock-kernel/lib/modules/5.15.0-mock \
    /mock-kernel/usr/share/initramfs-tools/hooks \
    /mock-kernel/usr/share/initramfs-tools/scripts \
    /mock-kernel/etc/initramfs-tools \
    /mock-kernel/boot \
    /solution

COPY mock-kernel/usr/share/initramfs-tools/hooks/sample-hook /mock-kernel/usr/share/initramfs-tools/hooks/sample-hook

RUN chmod +x /mock-kernel/usr/share/initramfs-tools/hooks/sample-hook

WORKDIR /workspace