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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages ansible

RUN mkdir -p /opt/ansible/inventory

COPY config/regions.conf /opt/ansible/inventory/regions.conf
COPY data/mock_hosts.json /opt/ansible/inventory/mock_hosts.json
COPY cloud_inventory.py /opt/ansible/inventory/cloud_inventory.py
COPY test_inventory.yml /opt/ansible/test_inventory.yml

RUN chmod +x /opt/ansible/inventory/cloud_inventory.py

WORKDIR /opt/ansible