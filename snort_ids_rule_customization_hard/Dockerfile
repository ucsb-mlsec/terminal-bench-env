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
    apt-get install -y --no-install-recommends \
    snort \
    libpcre3-dev \
    libdumbnet-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/snort /data /workspace

COPY data/network_traffic_logs.txt /data/network_traffic_logs.txt
COPY data/false_positive_incidents.txt /data/false_positive_incidents.txt
COPY data/current_ruleset.rules /data/current_ruleset.rules
COPY data/attack_coverage_requirements.txt /data/attack_coverage_requirements.txt

WORKDIR /workspace