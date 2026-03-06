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
    curl \
    gnupg \
    iproute2 \
    iptables \
    ca-certificates && \
    curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.noarmor.gpg | tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null && \
    curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/noble.tailscale-keyring.list | tee /etc/apt/sources.list.d/tailscale.list && \
    apt-get update && \
    apt-get install -y tailscale && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y iptables iproute2 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /peers/peer-alpha/etc/default && \
    mkdir -p /peers/peer-alpha/var/lib/tailscale && \
    mkdir -p /peers/peer-beta/etc/default && \
    mkdir -p /peers/peer-beta/var/lib/tailscale && \
    mkdir -p /peers/peer-gamma/etc/default && \
    mkdir -p /peers/peer-gamma/var/lib/tailscale

COPY peers/peer-alpha/etc/default/tailscaled /peers/peer-alpha/etc/default/tailscaled
COPY peers/peer-beta/etc/default/tailscaled /peers/peer-beta/etc/default/tailscaled
COPY peers/peer-gamma/etc/default/tailscaled /peers/peer-gamma/etc/default/tailscaled
COPY peers/peer-alpha/var/lib/tailscale/tailscaled.state /peers/peer-alpha/var/lib/tailscale/tailscaled.state
COPY peers/peer-beta/var/lib/tailscale/tailscaled.state /peers/peer-beta/var/lib/tailscale/tailscaled.state
COPY peers/peer-gamma/var/lib/tailscale/tailscaled.state /peers/peer-gamma/var/lib/tailscale/tailscaled.state

WORKDIR /root

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
