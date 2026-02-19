FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/bootloader_configs /opt/data

COPY config/config_01.conf /opt/bootloader_configs/config_01.conf
COPY config/config_02.conf /opt/bootloader_configs/config_02.conf
COPY config/config_03.conf /opt/bootloader_configs/config_03.conf
COPY config/config_04.conf /opt/bootloader_configs/config_04.conf
COPY data/shim_empty.efi /opt/data/shim_empty.efi
COPY data/shim_valid.efi /opt/data/shim_valid.efi
COPY data/shim_alt.efi /opt/data/shim_alt.efi
COPY data/cert_01.pem /opt/data/cert_01.pem
COPY data/cert_02.pem /opt/data/cert_02.pem
COPY data/cert_valid.pem /opt/data/cert_valid.pem
COPY data/cert_invalid.pem /opt/data/cert_invalid.pem

WORKDIR /opt/bootloader_configs