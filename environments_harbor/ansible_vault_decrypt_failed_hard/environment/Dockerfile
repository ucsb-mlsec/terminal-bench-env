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
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages ansible

RUN mkdir -p /workspace/config/group_vars /workspace/config/host_vars

COPY deploy.yml /workspace/deploy.yml
COPY config/group_vars/webservers.yml /workspace/config/group_vars/webservers.yml
COPY config/group_vars/dbservers.yml /workspace/config/group_vars/dbservers.yml
COPY config/host_vars/app01.yml /workspace/config/host_vars/app01.yml
COPY config/host_vars/app02.yml /workspace/config/host_vars/app02.yml
COPY config/group_vars/loadbalancers.yml /workspace/config/group_vars/loadbalancers.yml
COPY config/host_vars/cache01.yml /workspace/config/host_vars/cache01.yml
COPY vault_pass_old.txt /workspace/vault_pass_old.txt
COPY vault_pass_mid.txt /workspace/vault_pass_mid.txt
COPY vault_pass_new.txt /workspace/vault_pass_new.txt
COPY ansible.cfg /workspace/ansible.cfg
COPY inventory /workspace/inventory

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
