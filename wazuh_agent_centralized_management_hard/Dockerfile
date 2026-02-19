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
    python3 \
    python3-pip \
    libxml2-utils && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages lxml

RUN mkdir -p /var/ossec/etc && \
    mkdir -p /etc/wazuh-manager/shared/default && \
    mkdir -p /etc/wazuh-manager/shared/webservers && \
    mkdir -p /etc/wazuh-manager/shared/databases && \
    mkdir -p /etc/wazuh-manager/shared/appservers && \
    mkdir -p /etc/wazuh-manager/shared/dmz && \
    mkdir -p /root

COPY data/client.keys /var/ossec/etc/client.keys
COPY config/shared_default_agent.conf /etc/wazuh-manager/shared/default/agent.conf
COPY config/shared_webservers_agent.conf /etc/wazuh-manager/shared/webservers/agent.conf
COPY config/shared_databases_agent.conf /etc/wazuh-manager/shared/databases/agent.conf
COPY config/shared_appservers_agent.conf /etc/wazuh-manager/shared/appservers/agent.conf
COPY config/shared_dmz_agent.conf /etc/wazuh-manager/shared/dmz/agent.conf
COPY config/group_assignments.json /etc/wazuh-manager/group_assignments.json
COPY config/group_definitions.json /etc/wazuh-manager/group_definitions.json

WORKDIR /root