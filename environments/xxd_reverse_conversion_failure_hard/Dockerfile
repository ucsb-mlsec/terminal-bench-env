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
    apt-get install -y xxd && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/corrupted_hexdumps /workspace/recovered_binaries

COPY data/corrupted_hexdumps/firmware_boot.hex /workspace/corrupted_hexdumps/firmware_boot.hex
COPY data/corrupted_hexdumps/firmware_main.hex /workspace/corrupted_hexdumps/firmware_main.hex
COPY data/corrupted_hexdumps/firmware_radio.hex /workspace/corrupted_hexdumps/firmware_radio.hex
COPY data/corrupted_hexdumps/firmware_usb.hex /workspace/corrupted_hexdumps/firmware_usb.hex
COPY data/corrupted_hexdumps/firmware_network.hex /workspace/corrupted_hexdumps/firmware_network.hex
COPY data/corrupted_hexdumps/firmware_crypto.hex /workspace/corrupted_hexdumps/firmware_crypto.hex

WORKDIR /workspace