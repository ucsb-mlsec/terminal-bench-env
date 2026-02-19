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
    apt-get install -y openssl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages cryptography

RUN mkdir -p /etc/openvpn /root /workspace/data

COPY config/server-main.conf /etc/openvpn/server-main.conf
COPY config/server-branch-office.conf /etc/openvpn/server-branch-office.conf
COPY config/server-remote-workers.conf /etc/openvpn/server-remote-workers.conf
COPY config/server-datacenter.conf /etc/openvpn/server-datacenter.conf

COPY data/main-ca.crt /workspace/data/main-ca.crt
COPY data/main-server.crt /workspace/data/main-server.crt
COPY data/main-dh.pem /workspace/data/main-dh.pem
COPY data/branch-ca.crt /workspace/data/branch-ca.crt
COPY data/branch-server.crt /workspace/data/branch-server.crt
COPY data/branch-dh.pem /workspace/data/branch-dh.pem
COPY data/branch-ta.key /workspace/data/branch-ta.key
COPY data/remote-ca.crt /workspace/data/remote-ca.crt
COPY data/remote-server.crt /workspace/data/remote-server.crt
COPY data/remote-dh.pem /workspace/data/remote-dh.pem
COPY data/remote-tc.key /workspace/data/remote-tc.key
COPY data/datacenter-ca.crt /workspace/data/datacenter-ca.crt
COPY data/datacenter-server.crt /workspace/data/datacenter-server.crt

WORKDIR /workspace