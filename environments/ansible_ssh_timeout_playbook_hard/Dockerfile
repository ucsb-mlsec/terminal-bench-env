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
    ansible \
    openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y ansible && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/admin/ansible/group_vars

COPY config/ansible.cfg /home/admin/ansible/ansible.cfg
COPY config/inventory.ini /home/admin/ansible/inventory.ini
COPY config/deploy.yml /home/admin/ansible/deploy.yml
COPY config/group_vars/production.yml /home/admin/ansible/group_vars/production.yml
COPY config/group_vars/staging.yml /home/admin/ansible/group_vars/staging.yml
COPY config/group_vars/development.yml /home/admin/ansible/group_vars/development.yml

WORKDIR /home/admin