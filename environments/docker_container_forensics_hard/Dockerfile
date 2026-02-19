FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directory structure for the container filesystem
RUN mkdir -p /home/agent/container_fs/var/spool/cron/crontabs && \
    mkdir -p /home/agent/container_fs/root/.ssh && \
    mkdir -p /home/agent/container_fs/app && \
    mkdir -p /home/agent/container_fs/etc && \
    mkdir -p /home/agent/container_fs/var/log && \
    mkdir -p /home/agent/container_fs/home/appuser && \
    mkdir -p /tmp

# Copy all generated files to their correct locations in the extracted container filesystem
COPY container_fs/var/spool/cron/crontabs/www-data /home/agent/container_fs/var/spool/cron/crontabs/www-data
COPY container_fs/root/.ssh/authorized_keys /home/agent/container_fs/root/.ssh/authorized_keys
COPY container_fs/app/server.py /home/agent/container_fs/app/server.py
COPY container_fs/etc/passwd /home/agent/container_fs/etc/passwd
COPY container_fs/etc/shadow /home/agent/container_fs/etc/shadow
COPY container_fs/app/config.py /home/agent/container_fs/app/config.py
COPY container_fs/app/requirements.txt /home/agent/container_fs/app/requirements.txt
COPY container_fs/var/log/auth.log /home/agent/container_fs/var/log/auth.log
COPY container_fs/var/log/syslog /home/agent/container_fs/var/log/syslog
COPY container_fs/home/appuser/.bash_history /home/agent/container_fs/home/appuser/.bash_history
COPY container_fs/etc/crontab /home/agent/container_fs/etc/crontab

WORKDIR /home/agent