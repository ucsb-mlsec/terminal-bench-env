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
    apt-get install -y ansible && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/ansible/inventory /opt/ansible/data /tmp

COPY data/servers.json /opt/ansible/data/servers.json
COPY opt/ansible/inventory/servers.py /opt/ansible/inventory/servers.py

RUN chmod +x /opt/ansible/inventory/servers.py

WORKDIR /opt/ansible