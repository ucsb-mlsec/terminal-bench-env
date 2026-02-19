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
    duplicity \
    gnupg \
    python3-boto \
    python3-s3transfer && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y duplicity gnupg && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/appdata /root /var/cache/duplicity

COPY data/appdata/sample1.txt /opt/appdata/sample1.txt
COPY data/appdata/config.json /opt/appdata/config.json
COPY data/appdata/database.db /opt/appdata/database.db
COPY config/.s3cfg /root/.s3cfg
COPY config/.backup_passphrase /root/.backup_passphrase

RUN chmod 600 /root/.s3cfg /root/.backup_passphrase

WORKDIR /root