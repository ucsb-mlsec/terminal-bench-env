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
    jq \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/docker \
    /root/.docker \
    /var/lib/docker \
    /usr/local/etc/docker \
    /tmp/daemon-configs \
    /root/solution

COPY config/etc_docker_daemon.json /etc/docker/daemon.json
COPY config/var_lib_docker_daemon.json /var/lib/docker/daemon.json
COPY config/usr_local_etc_docker_daemon.json /usr/local/etc/docker/daemon.json
COPY config/etc_docker_daemon.json.bak /etc/docker/daemon.json.bak
COPY config/etc_docker_daemon.json.old /etc/docker/daemon.json.old
COPY config/tmp_daemon_configs_daemon.json /tmp/daemon-configs/daemon.json

WORKDIR /root