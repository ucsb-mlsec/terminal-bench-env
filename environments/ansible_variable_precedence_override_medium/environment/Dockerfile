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
    apt-get install -y software-properties-common && \
    add-apt-repository --yes --update ppa:ansible/ansible && \
    apt-get install -y ansible && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages ansible-core>=2.12

RUN mkdir -p /workspace/config/group_vars \
    /workspace/config/host_vars \
    /workspace/config/roles/webapp/defaults \
    /workspace/config/roles/webapp/vars

COPY config/group_vars/webservers.yml /workspace/config/group_vars/webservers.yml
COPY config/host_vars/prod-web-01.yml /workspace/config/host_vars/prod-web-01.yml
COPY config/roles/webapp/defaults/main.yml /workspace/config/roles/webapp/defaults/main.yml
COPY config/roles/webapp/vars/main.yml /workspace/config/roles/webapp/vars/main.yml

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
