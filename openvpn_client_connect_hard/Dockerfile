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
    apt-get install -y openvpn && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/vpn/configs /opt/vpn/certs /root

COPY config/client1.conf /opt/vpn/configs/client1.conf
COPY config/client2.ovpn /opt/vpn/configs/client2.ovpn
COPY config/client3.conf /opt/vpn/configs/client3.conf
COPY config/client4.ovpn /opt/vpn/configs/client4.ovpn
COPY config/client5.conf /opt/vpn/configs/client5.conf

COPY data/ca.crt /opt/vpn/certs/ca.crt
COPY data/client1-cert.crt /opt/vpn/certs/client1-cert.crt
COPY data/client1-key.pem /opt/vpn/certs/client1-key.pem
COPY data/client2.crt /opt/vpn/certs/client2.crt
COPY data/client2.key /opt/vpn/certs/client2.key
COPY data/client4.crt /opt/vpn/certs/client4.crt
COPY data/client4.key /opt/vpn/certs/client4.key

WORKDIR /root