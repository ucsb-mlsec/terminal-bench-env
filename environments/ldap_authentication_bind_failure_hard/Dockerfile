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
    apt-get install -y slapd ldap-utils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/ldap/certs /var/log/slapd /tmp

COPY config/slapd.conf /etc/ldap/slapd.conf
COPY config/certs/server.crt /etc/ldap/certs/server.crt
COPY config/certs/ca.crt /etc/ldap/certs/ca.crt
COPY data/slapd.log /var/log/slapd/slapd.log

WORKDIR /tmp