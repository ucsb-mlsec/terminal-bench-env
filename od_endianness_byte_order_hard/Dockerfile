FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /firmware/configs /firmware/specs /solution

COPY data/firmware/specs/device_specs.txt /firmware/specs/device_specs.txt
COPY data/firmware/configs/device_A.bin /firmware/configs/device_A.bin
COPY data/firmware/configs/device_B.bin /firmware/configs/device_B.bin
COPY data/firmware/configs/device_C.bin /firmware/configs/device_C.bin
COPY data/firmware/configs/device_D.bin /firmware/configs/device_D.bin
COPY data/firmware/configs/device_E.bin /firmware/configs/device_E.bin

WORKDIR /workspace