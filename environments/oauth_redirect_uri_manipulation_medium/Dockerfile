FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/log/oauth \
    && mkdir -p /home/investigator/app_config \
    && mkdir -p /home/investigator/network_capture \
    && mkdir -p /home/investigator

COPY data/auth_requests.log /var/log/oauth/auth_requests.log
COPY data/token_exchanges.log /var/log/oauth/token_exchanges.log
COPY data/requests.pcap /home/investigator/network_capture/requests.pcap

WORKDIR /home/investigator