FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/alice/.ssh \
    /home/bob/.ssh \
    /home/carol/.ssh \
    /home/carol/backup_keys \
    /home/dave/.ssh \
    /home/eve/.ssh \
    /home/frank/.ssh \
    /opt/keys \
    /root/.ssh \
    /var/backups/keys \
    /tmp

COPY home/alice/.ssh/authorized_keys /home/alice/.ssh/authorized_keys
COPY home/alice/.ssh/id_rsa /home/alice/.ssh/id_rsa
COPY home/alice/.ssh/id_ed25519 /home/alice/.ssh/id_ed25519
COPY home/bob/.ssh/id_rsa /home/bob/.ssh/id_rsa
COPY home/bob/.ssh/config /home/bob/.ssh/config
COPY home/carol/.ssh/authorized_keys /home/carol/.ssh/authorized_keys
COPY home/carol/backup_keys/id_rsa /home/carol/backup_keys/id_rsa
COPY home/dave/.ssh/id_dsa /home/dave/.ssh/id_dsa
COPY home/dave/.ssh/authorized_keys /home/dave/.ssh/authorized_keys
COPY opt/keys/bob_key /opt/keys/bob_key
COPY opt/keys/deploy_key /opt/keys/deploy_key
COPY root/.ssh/id_rsa /root/.ssh/id_rsa
COPY root/.ssh/authorized_keys /root/.ssh/authorized_keys
COPY var/backups/keys/old_id_rsa /var/backups/keys/old_id_rsa
COPY home/eve/.ssh/id_ecdsa /home/eve/.ssh/id_ecdsa
COPY home/frank/.ssh/authorized_keys /home/frank/.ssh/authorized_keys

WORKDIR /tmp