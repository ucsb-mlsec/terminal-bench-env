FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/legacy_backups/level1/level2/level3 \
    /opt/legacy_backups/restricted \
    /opt/legacy_backups/.hidden_dir \
    /opt/legacy_backups/empty_looking \
    /home/agent

COPY opt/legacy_backups/.hidden_log /opt/legacy_backups/.hidden_log
COPY opt/legacy_backups/.system_state /opt/legacy_backups/.system_state
COPY opt/legacy_backups/level1/.metadata /opt/legacy_backups/level1/.metadata
COPY opt/legacy_backups/level1/backup_data.txt /opt/legacy_backups/level1/backup_data.txt
COPY opt/legacy_backups/level1/level2/.checkpoint /opt/legacy_backups/level1/level2/.checkpoint
COPY opt/legacy_backups/level1/level2/archive.dat /opt/legacy_backups/level1/level2/archive.dat
COPY opt/legacy_backups/level1/level2/level3/.marker /opt/legacy_backups/level1/level2/level3/.marker
COPY opt/legacy_backups/level1/level2/level3/deep_file.txt /opt/legacy_backups/level1/level2/level3/deep_file.txt
COPY opt/legacy_backups/restricted/.protected /opt/legacy_backups/restricted/.protected
COPY opt/legacy_backups/restricted/secure_backup.txt /opt/legacy_backups/restricted/secure_backup.txt
COPY opt/legacy_backups/.hidden_dir/.nested_hidden /opt/legacy_backups/.hidden_dir/.nested_hidden
COPY opt/legacy_backups/empty_looking/README.txt /opt/legacy_backups/empty_looking/README.txt

RUN chmod -R 000 /opt/legacy_backups/restricted

WORKDIR /home/agent