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
    apt-get install -y \
    ansible \
    sshpass \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y sshpass && \
    pip3 install --no-cache-dir --break-system-packages ansible>=2.9 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages ansible-core

RUN mkdir -p /opt/ansible-project/group_vars

COPY config/site.yml /opt/ansible-project/site.yml
COPY config/inventory.ini /opt/ansible-project/inventory.ini
COPY config/group_vars/all.yml /opt/ansible-project/group_vars/all.yml
COPY config/vault.yml /opt/ansible-project/vault.yml

WORKDIR /opt/ansible-project