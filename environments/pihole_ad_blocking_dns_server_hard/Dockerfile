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
    apt-get install -y dnsmasq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/dnsmasq.d /root

COPY etc/dnsmasq.d/blocklist_ads.conf /etc/dnsmasq.d/blocklist_ads.conf
COPY etc/dnsmasq.d/blocklist_tracking.conf /etc/dnsmasq.d/blocklist_tracking.conf
COPY etc/dnsmasq.d/blocklist_social.conf /etc/dnsmasq.d/blocklist_social.conf
COPY etc/dnsmasq.d/old_blocks.conf /etc/dnsmasq.d/old_blocks.conf

WORKDIR /root