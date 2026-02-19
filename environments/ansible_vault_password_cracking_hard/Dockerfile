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
    apt-get install -y ansible && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages ansible pyyaml

RUN mkdir -p /data/prod /data/staging /data/dev /data/backup /solution

COPY data/prod/database.vault /data/prod/database.vault
COPY data/prod/api_keys.vault /data/prod/api_keys.vault
COPY data/staging/secrets.vault /data/staging/secrets.vault
COPY data/staging/app_config.vault /data/staging/app_config.vault
COPY data/dev/credentials.vault /data/dev/credentials.vault
COPY data/dev/test_vault.vault /data/dev/test_vault.vault
COPY data/backup/old_secrets.vault /data/backup/old_secrets.vault
COPY data/prod/README.txt /data/prod/README.txt
COPY data/staging/notes.txt /data/staging/notes.txt
COPY data/dev/info.txt /data/dev/info.txt
COPY data/wordlist.txt /data/wordlist.txt
COPY data/config.yaml /data/config.yaml
COPY data/inventory.ini /data/inventory.ini

WORKDIR /data