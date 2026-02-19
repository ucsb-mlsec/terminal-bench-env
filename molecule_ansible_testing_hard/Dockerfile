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
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages ansible>=2.9 jinja2>=2.11 pyyaml>=5.3

RUN mkdir -p /workspace/ansible-role-webserver/tasks \
    /workspace/ansible-role-webserver/handlers \
    /workspace/ansible-role-webserver/defaults \
    /workspace/ansible-role-webserver/vars \
    /workspace/ansible-role-webserver/templates \
    /workspace/ansible-role-webserver/meta

COPY ansible-role-webserver/tasks/main.yml /workspace/ansible-role-webserver/tasks/main.yml
COPY ansible-role-webserver/tasks/debian.yml /workspace/ansible-role-webserver/tasks/debian.yml
COPY ansible-role-webserver/tasks/redhat.yml /workspace/ansible-role-webserver/tasks/redhat.yml
COPY ansible-role-webserver/handlers/main.yml /workspace/ansible-role-webserver/handlers/main.yml
COPY ansible-role-webserver/defaults/main.yml /workspace/ansible-role-webserver/defaults/main.yml
COPY ansible-role-webserver/vars/main.yml /workspace/ansible-role-webserver/vars/main.yml
COPY ansible-role-webserver/templates/vhost.conf.j2 /workspace/ansible-role-webserver/templates/vhost.conf.j2
COPY ansible-role-webserver/templates/nginx.conf.j2 /workspace/ansible-role-webserver/templates/nginx.conf.j2
COPY ansible-role-webserver/meta/main.yml /workspace/ansible-role-webserver/meta/main.yml

WORKDIR /workspace