FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/ansible_env /home/agent

COPY data/ansible_env/inventory.ini /home/agent/ansible_env/inventory.ini
COPY data/ansible_env/ansible.cfg /home/agent/ansible_env/ansible.cfg
COPY data/ansible_env/ssh_config_web1 /home/agent/ansible_env/ssh_config_web1
COPY data/ansible_env/ssh_config_web2 /home/agent/ansible_env/ssh_config_web2
COPY data/ansible_env/ssh_config_db1 /home/agent/ansible_env/ssh_config_db1
COPY data/ansible_env/ssh_config_db2 /home/agent/ansible_env/ssh_config_db2
COPY data/ansible_env/ssh_config_app1 /home/agent/ansible_env/ssh_config_app1
COPY data/ansible_env/ssh_config_app2 /home/agent/ansible_env/ssh_config_app2
COPY data/ansible_env/ssh_config_monitor1 /home/agent/ansible_env/ssh_config_monitor1
COPY data/ansible_env/ssh_config_backup1 /home/agent/ansible_env/ssh_config_backup1
COPY data/ansible_env/network_tests.json /home/agent/ansible_env/network_tests.json

WORKDIR /home/agent