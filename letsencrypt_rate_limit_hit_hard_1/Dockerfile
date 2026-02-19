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
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y jq certbot && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages python-dateutil

RUN mkdir -p /opt/certs /opt/scripts /var/log/certbot /tmp

COPY data/cert_inventory.json /opt/certs/cert_inventory.json
COPY data/certbot_renewal.log /var/log/certbot/certbot_renewal.log
COPY renew_certs.sh /opt/scripts/renew_certs.sh

RUN chmod +x /opt/scripts/renew_certs.sh

WORKDIR /workspace