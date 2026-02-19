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
    bash \
    coreutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/data /workspace/config

COPY packer-template.json /workspace/
COPY setup-base.sh /workspace/
COPY install-packages.sh /workspace/
COPY configure-services.sh /workspace/
COPY cleanup.sh /workspace/
COPY verify-install.sh /workspace/
COPY data/build-log-001.txt /workspace/data/
COPY data/build-log-002.txt /workspace/data/
COPY data/build-log-003.txt /workspace/data/
COPY config/build-vars.json /workspace/config/

RUN chmod +x /workspace/*.sh

WORKDIR /workspace