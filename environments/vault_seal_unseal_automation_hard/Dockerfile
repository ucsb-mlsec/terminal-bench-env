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
    apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/vault/state/unseal_keys /opt/vault /tmp

COPY data/seal_status.json /var/vault/state/seal_status.json
COPY data/key2.txt /var/vault/state/unseal_keys/key2.txt
COPY data/key3.txt /var/vault/state/unseal_keys/key3.txt

RUN echo "unseal_key_1" > /var/vault/state/unseal_keys/key1.txt

WORKDIR /opt/vault