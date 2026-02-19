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
    tshark \
    tcpdump \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y tshark wireshark-common && \
    pip3 install --no-cache-dir --break-system-packages scapy pyshark && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    scapy \
    pyshark

RUN mkdir -p /data

COPY data/jenkins_traffic.pcap /data/jenkins_traffic.pcap
COPY data/jenkins.log /data/jenkins.log
COPY data/audit.log /data/audit.log
COPY data/config.xml /data/config.xml

WORKDIR /workspace