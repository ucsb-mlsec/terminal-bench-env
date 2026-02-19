FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/lib/dpkg /usr/bin /bin /etc /root

COPY var/lib/dpkg/diversions /var/lib/dpkg/diversions
COPY var/lib/dpkg/status /var/lib/dpkg/status
COPY usr/bin/editor.distrib /usr/bin/editor.distrib
COPY bin/sh.distrib /bin/sh.distrib
COPY usr/bin/vim.distrib /usr/bin/vim.distrib
COPY etc/config.distrib /etc/config.distrib

WORKDIR /root