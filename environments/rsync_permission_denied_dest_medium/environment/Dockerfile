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
    apt-get install -y rsync && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/webapp/data /backup/webapp /home/user && \
    useradd -m -s /bin/bash backupuser && \
    useradd -m -s /bin/bash webapp && \
    useradd -m -s /bin/bash www-data 2>/dev/null || true && \
    useradd -m -s /bin/bash user 2>/dev/null || true

COPY data/webapp_files.txt /opt/webapp/data/webapp_files.txt

RUN chown -R webapp:webapp /opt/webapp/data && \
    chmod 755 /opt/webapp/data

WORKDIR /home/user

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
