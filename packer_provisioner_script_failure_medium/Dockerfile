FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install Packer dependencies and Packer itself
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    unzip \
    gpg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/hashicorp.list && \
    apt-get update && \
    apt-get install -y packer && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/packer-build/scripts

COPY opt/packer-build/template.json /opt/packer-build/template.json
COPY opt/packer-build/scripts/update-system.sh /opt/packer-build/scripts/update-system.sh
COPY opt/packer-build/scripts/install-docker.sh /opt/packer-build/scripts/install-docker.sh
COPY opt/packer-build/scripts/configure-network.sh /opt/packer-build/scripts/configure-network.sh
COPY opt/packer-build/scripts/cleanup.sh /opt/packer-build/scripts/cleanup.sh

WORKDIR /opt/packer-build