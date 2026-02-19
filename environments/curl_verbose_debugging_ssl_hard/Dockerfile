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
    apt-get install -y curl openssl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/server /opt/certs /root

COPY opt/server/config.py /opt/server/config.py
COPY opt/server/restart.sh /opt/server/restart.sh
COPY opt/server/server.py /opt/server/server.py
COPY opt/certs/server.pem /opt/certs/server.pem
COPY opt/certs/server-key.pem /opt/certs/server-key.pem
COPY opt/certs/intermediate.pem /opt/certs/intermediate.pem
COPY opt/certs/root-ca.pem /opt/certs/root-ca.pem

RUN chmod +x /opt/server/restart.sh

WORKDIR /root