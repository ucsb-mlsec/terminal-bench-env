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
    apt-get install -y curl jq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/vault /etc/vault /secure /var/log && \
    chmod 755 /opt/vault /etc/vault && \
    chmod 700 /secure && \
    chmod 755 /var/log

COPY data/vault-keys.enc /secure/vault-keys.enc

RUN chmod 600 /secure/vault-keys.enc

RUN echo '{ \
  "nodes": [ \
    {"name": "vault-01", "address": "https://vault-01.internal:8200"}, \
    {"name": "vault-02", "address": "https://vault-02.internal:8200"}, \
    {"name": "vault-03", "address": "https://vault-03.internal:8200"} \
  ], \
  "unseal_keys_file": "/secure/vault-keys.enc", \
  "required_shares": 3, \
  "timeout_seconds": 300 \
}' > /etc/vault/config.json

WORKDIR /opt/vault