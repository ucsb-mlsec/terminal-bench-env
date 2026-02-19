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
    dkms \
    build-essential \
    linux-headers-$(uname -r) \
    kmod && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y dkms build-essential linux-headers-generic kmod && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/src/netmon-1.0.0

COPY usr/src/netmon-1.0.0/netmon.c /usr/src/netmon-1.0.0/netmon.c
COPY usr/src/netmon-1.0.0/Makefile /usr/src/netmon-1.0.0/Makefile
COPY usr/src/netmon-1.0.0/dkms.conf /usr/src/netmon-1.0.0/dkms.conf

WORKDIR /root