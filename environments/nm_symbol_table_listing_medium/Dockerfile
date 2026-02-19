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
    apt-get install -y binutils file && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/legacy_libs /tmp

COPY opt/legacy_libs/libmath.so /opt/legacy_libs/libmath.so
COPY opt/legacy_libs/libcrypto_legacy.so /opt/legacy_libs/libcrypto_legacy.so
COPY opt/legacy_libs/libconfig.so /opt/legacy_libs/libconfig.so
COPY opt/legacy_libs/libutils.so /opt/legacy_libs/libutils.so
COPY opt/legacy_libs/libnetwork.so /opt/legacy_libs/libnetwork.so
COPY opt/legacy_libs/liblogger.so /opt/legacy_libs/liblogger.so

WORKDIR /workspace