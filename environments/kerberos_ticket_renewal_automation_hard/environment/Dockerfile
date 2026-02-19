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
    apt-get install -y \
    cron \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data /config /solution

COPY data/token_db.txt /data/token_db.txt
COPY data/token_api.txt /data/token_api.txt
COPY data/token_cache.txt /data/token_cache.txt
COPY data/token_storage.txt /data/token_storage.txt
COPY data/token_messaging.txt /data/token_messaging.txt
COPY config/renewal_credentials.conf /config/renewal_credentials.conf
COPY config/service_mappings.conf /config/service_mappings.conf
COPY config/legacy_creds.conf /config/legacy_creds.conf

RUN chmod 644 /data/*.txt && \
    chmod 644 /config/*.conf

WORKDIR /solution

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
