FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/pbs_configs /home/agent

COPY config/pbs.conf /opt/pbs_configs/pbs.conf
COPY config/pbs_backup.conf /opt/pbs_configs/pbs_backup.conf
COPY config/server_priv_nodes /opt/pbs_configs/server_priv_nodes
COPY config/pbs_environment /opt/pbs_configs/pbs_environment
COPY config/server_config.txt /opt/pbs_configs/server_config.txt

WORKDIR /home/agent