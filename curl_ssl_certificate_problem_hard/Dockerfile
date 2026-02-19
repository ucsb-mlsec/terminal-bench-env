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
    apt-get install -y curl ca-certificates openssl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/certificates /etc/ssl/certs /root /data

COPY data/server.crt /data/server.crt
COPY data/server.key /data/server.key
COPY data/custom-ca.crt /data/custom-ca.crt
COPY data/random-cert1.crt /data/random-cert1.crt
COPY data/old-ca.pem /data/old-ca.pem
COPY data/intermediate.crt /data/intermediate.crt

COPY data/server.crt /opt/certificates/server.crt
COPY data/server.key /opt/certificates/server.key
COPY data/custom-ca.crt /opt/certificates/custom-ca.crt
COPY data/random-cert1.crt /opt/certificates/random-cert1.crt
COPY data/old-ca.pem /opt/certificates/old-ca.pem
COPY data/intermediate.crt /opt/certificates/intermediate.crt

RUN pip3 install --no-cache-dir --break-system-packages flask

WORKDIR /root