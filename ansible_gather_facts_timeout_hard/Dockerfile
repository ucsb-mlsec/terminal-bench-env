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
    && add-apt-repository --yes --update ppa:ansible/ansible && \
    apt-get install -y --no-install-recommends \
    ansible \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/ansible/config/inventory && \
    mkdir -p /home/ansible/group_vars/legacy_systems && \
    mkdir -p /home/ansible/group_vars/remote_sites && \
    mkdir -p /home/ansible/group_vars/edge_devices && \
    mkdir -p /home/ansible/group_vars/production_servers

COPY config/ansible.cfg /home/ansible/config/ansible.cfg
COPY config/inventory/hosts /home/ansible/config/inventory/hosts
COPY test_playbook.yml /home/ansible/test_playbook.yml

WORKDIR /home/ansible