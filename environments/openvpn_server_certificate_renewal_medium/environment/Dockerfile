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

RUN mkdir -p /etc/openvpn/easy-rsa/pki/issued

COPY etc/openvpn/easy-rsa/pki/ca.crt /etc/openvpn/easy-rsa/pki/ca.crt
COPY etc/openvpn/easy-rsa/pki/issued/vpn-server.crt /etc/openvpn/easy-rsa/pki/issued/vpn-server.crt
COPY etc/openvpn/easy-rsa/pki/issued/client1.crt /etc/openvpn/easy-rsa/pki/issued/client1.crt
COPY etc/openvpn/easy-rsa/pki/issued/client2.crt /etc/openvpn/easy-rsa/pki/issued/client2.crt

WORKDIR /root

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
