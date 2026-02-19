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
    apt-get install -y curl wget unzip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages hvac requests

RUN wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip && \
    unzip vault_1.15.0_linux_amd64.zip && \
    mv vault /usr/local/bin/ && \
    rm vault_1.15.0_linux_amd64.zip

RUN mkdir -p /tmp

COPY data/db_config.txt /tmp/db_config.txt
COPY app.py /tmp/app.py

WORKDIR /tmp