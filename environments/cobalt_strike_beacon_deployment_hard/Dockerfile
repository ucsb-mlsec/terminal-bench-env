FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /incident/staging_server /solution

COPY data/staging_server/index.html /incident/staging_server/index.html
COPY data/staging_server/payload.exe /incident/staging_server/payload.exe
COPY data/staging_server/update.js /incident/staging_server/update.js
COPY data/staging_server/beacon.conf /incident/staging_server/beacon.conf
COPY data/staging_server/readme.txt /incident/staging_server/readme.txt
COPY data/staging_server/data.dat /incident/staging_server/data.dat

WORKDIR /incident