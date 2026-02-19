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
RUN pip3 install --no-cache-dir --break-system-packages ansible>=2.9

RUN pip3 install --no-cache-dir --break-system-packages jinja2

RUN mkdir -p /ansible-project/roles/webapp/defaults \
    /ansible-project/roles/webapp/vars \
    /ansible-project/roles/webapp/templates

COPY ansible-project/site.yml /ansible-project/site.yml
COPY ansible-project/roles/webapp/defaults/main.yml /ansible-project/roles/webapp/defaults/main.yml
COPY ansible-project/roles/webapp/vars/main.yml /ansible-project/roles/webapp/vars/main.yml
COPY ansible-project/roles/webapp/templates/nginx.conf.j2 /ansible-project/roles/webapp/templates/nginx.conf.j2
COPY ansible-project/roles/webapp/templates/site.conf.j2 /ansible-project/roles/webapp/templates/site.conf.j2
COPY ansible-project/roles/webapp/templates/upstream.conf.j2 /ansible-project/roles/webapp/templates/upstream.conf.j2
COPY ansible-project/roles/webapp/templates/proxy.conf.j2 /ansible-project/roles/webapp/templates/proxy.conf.j2
COPY ansible-project/validate.sh /ansible-project/validate.sh

RUN chmod +x /ansible-project/validate.sh

WORKDIR /ansible-project

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
