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
    apt-get install -y bind9-utils && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages dnspython

RUN mkdir -p /var/dns/zones /tmp

COPY data/example.com.zone /var/dns/zones/example.com.zone
COPY data/secure.zone /var/dns/zones/secure.zone
COPY data/broken-serial.zone /var/dns/zones/broken-serial.zone
COPY data/wildcard.zone /var/dns/zones/wildcard.zone

WORKDIR /workspace