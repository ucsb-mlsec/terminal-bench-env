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
    apt-get install -y sqlite3 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/backups/webapp/public/images && \
    mkdir -p /opt/webapp/data && \
    mkdir -p /opt/webapp/config && \
    mkdir -p /opt/webapp/public && \
    useradd -r -s /bin/false webapp || true

COPY data/var/backups/webapp/app.db /var/backups/webapp/app.db
COPY data/var/backups/webapp/settings.conf /var/backups/webapp/settings.conf
COPY data/var/backups/webapp/public/index.html /var/backups/webapp/public/index.html
COPY data/var/backups/webapp/public/style.css /var/backups/webapp/public/style.css
COPY data/var/backups/webapp/public/app.js /var/backups/webapp/public/app.js
COPY data/var/backups/webapp/public/images/logo.svg /var/backups/webapp/public/images/logo.svg

WORKDIR /root