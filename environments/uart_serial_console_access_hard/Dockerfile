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
    john \
    john-data \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/assessment /tmp

COPY opt/assessment/boot.log /opt/assessment/boot.log
COPY opt/assessment/shadow /opt/assessment/shadow
COPY opt/assessment/inittab /opt/assessment/inittab
COPY opt/assessment/bootargs.conf /opt/assessment/bootargs.conf
COPY opt/assessment/login.conf /opt/assessment/login.conf

WORKDIR /opt/assessment