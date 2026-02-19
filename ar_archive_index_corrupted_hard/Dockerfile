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
    apt-get install -y binutils build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/legacy_project/lib && \
    mkdir -p /workspace/legacy_project/drivers && \
    mkdir -p /workspace/legacy_project/network && \
    mkdir -p /workspace/legacy_project/hardware/gpio && \
    mkdir -p /workspace/legacy_project/middleware && \
    mkdir -p /workspace/legacy_project/build/temp && \
    mkdir -p /workspace/legacy_project/protocols

COPY workspace/legacy_project/lib/libcore.a /workspace/legacy_project/lib/libcore.a
COPY workspace/legacy_project/lib/libutils.a /workspace/legacy_project/lib/libutils.a
COPY workspace/legacy_project/drivers/libserial.a /workspace/legacy_project/drivers/libserial.a
COPY workspace/legacy_project/network/libnetwork.a /workspace/legacy_project/network/libnetwork.a
COPY workspace/legacy_project/hardware/gpio/libgpio.a /workspace/legacy_project/hardware/gpio/libgpio.a
COPY workspace/legacy_project/middleware/libmiddleware.a /workspace/legacy_project/middleware/libmiddleware.a
COPY workspace/legacy_project/build/temp/libtemp.a /workspace/legacy_project/build/temp/libtemp.a
COPY workspace/legacy_project/protocols/libprotocol.a /workspace/legacy_project/protocols/libprotocol.a

WORKDIR /workspace