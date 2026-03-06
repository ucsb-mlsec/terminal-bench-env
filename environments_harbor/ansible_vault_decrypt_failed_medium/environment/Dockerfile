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
RUN pip3 install --no-cache-dir --break-system-packages ansible

RUN mkdir -p /workspace/config \
    /workspace/vars \
    /workspace/group_vars/webservers \
    /workspace/group_vars/databases \
    /workspace/host_vars/db01 \
    /workspace/host_vars/web01 \
    /workspace/roles/common/vars

COPY config/vault_password_1.txt /workspace/config/vault_password_1.txt
COPY config/vault_password_2.txt /workspace/config/vault_password_2.txt
COPY config/vault_password_3.txt /workspace/config/vault_password_3.txt
COPY config/vault_password_standard.txt /workspace/config/vault_password_standard.txt
COPY vars/prod_db.yml /workspace/vars/prod_db.yml
COPY vars/api_keys.yml /workspace/vars/api_keys.yml
COPY group_vars/webservers/vault.yml /workspace/group_vars/webservers/vault.yml
COPY group_vars/databases/secrets.yml /workspace/group_vars/databases/secrets.yml
COPY host_vars/db01/secrets.yml /workspace/host_vars/db01/secrets.yml
COPY host_vars/web01/vault.yml /workspace/host_vars/web01/vault.yml
COPY roles/common/vars/vault_secrets.yml /workspace/roles/common/vars/vault_secrets.yml
COPY deploy_production.yml /workspace/deploy_production.yml

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
