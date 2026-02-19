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
    wireguard-tools \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y wireguard-tools && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/vpn_backup /solution /workspace

COPY config/site_a.conf /opt/vpn_backup/site_a.conf
COPY config/site_b.conf /opt/vpn_backup/site_b.conf
COPY config/site_c.conf /opt/vpn_backup/site_c.conf
COPY config/site_d.conf /opt/vpn_backup/site_d.conf
COPY config/site_e_notes.txt /opt/vpn_backup/site_e_notes.txt
COPY config/keys_backup.txt /opt/vpn_backup/keys_backup.txt
COPY config/endpoints.txt /opt/vpn_backup/endpoints.txt

WORKDIR /workspace