FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/route53 \
    /var/log/app \
    /etc/network \
    /opt/certs \
    /tmp

COPY config/health_check_config.json /opt/route53/health_check_config.json
COPY data/access.log /var/log/app/access.log
COPY data/firewall_rules.conf /etc/network/firewall_rules.conf
COPY data/network_config.conf /etc/network/network_config.conf
COPY data/cert_info.txt /opt/certs/cert_info.txt

WORKDIR /tmp