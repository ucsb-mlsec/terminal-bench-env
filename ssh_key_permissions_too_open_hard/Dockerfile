FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/sshuser/.ssh/backup_keys /home/sshuser/.ssh/legacy_keys /tmp

COPY home/sshuser/.ssh/id_rsa_prod /home/sshuser/.ssh/id_rsa_prod
COPY home/sshuser/.ssh/id_rsa_prod.pub /home/sshuser/.ssh/id_rsa_prod.pub
COPY home/sshuser/.ssh/id_rsa_staging /home/sshuser/.ssh/id_rsa_staging
COPY home/sshuser/.ssh/id_rsa_staging.pub /home/sshuser/.ssh/id_rsa_staging.pub
COPY home/sshuser/.ssh/id_ed25519_dev /home/sshuser/.ssh/id_ed25519_dev
COPY home/sshuser/.ssh/id_ed25519_dev.pub /home/sshuser/.ssh/id_ed25519_dev.pub
COPY home/sshuser/.ssh/backup_keys/id_rsa_backup1 /home/sshuser/.ssh/backup_keys/id_rsa_backup1
COPY home/sshuser/.ssh/backup_keys/id_rsa_backup2 /home/sshuser/.ssh/backup_keys/id_rsa_backup2
COPY home/sshuser/.ssh/legacy_keys/id_dsa_old /home/sshuser/.ssh/legacy_keys/id_dsa_old
COPY home/sshuser/.ssh/config /home/sshuser/.ssh/config
COPY home/sshuser/.ssh/authorized_keys /home/sshuser/.ssh/authorized_keys
COPY home/sshuser/.ssh/known_hosts /home/sshuser/.ssh/known_hosts
COPY home/sshuser/.ssh/environment /home/sshuser/.ssh/environment

RUN useradd -m -s /bin/bash sshuser || true

RUN chmod 0644 /home/sshuser/.ssh/id_rsa_prod && \
    chmod 0777 /home/sshuser/.ssh/id_rsa_staging && \
    chmod 0755 /home/sshuser/.ssh/id_ed25519_dev && \
    chmod 0644 /home/sshuser/.ssh/backup_keys/id_rsa_backup1 && \
    chmod 0666 /home/sshuser/.ssh/backup_keys/id_rsa_backup2 && \
    chmod 0755 /home/sshuser/.ssh/legacy_keys/id_dsa_old && \
    chmod 0644 /home/sshuser/.ssh/config && \
    chmod 0755 /home/sshuser/.ssh && \
    chmod 0777 /home/sshuser/.ssh/backup_keys && \
    chmod 0750 /home/sshuser/.ssh/legacy_keys && \
    chmod 0644 /home/sshuser/.ssh/authorized_keys && \
    chmod 0666 /home/sshuser/.ssh/known_hosts && \
    chmod 0640 /home/sshuser/.ssh/environment && \
    chown -R sshuser:sshuser /home/sshuser/.ssh

WORKDIR /home/sshuser