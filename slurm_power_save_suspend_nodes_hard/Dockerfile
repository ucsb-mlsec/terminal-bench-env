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
    slurm-wlm \
    munge \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/slurm /usr/local/sbin /root

COPY config/slurm.conf /etc/slurm/slurm.conf
COPY config/nodes.conf /etc/slurm/nodes.conf
COPY config/power_save.conf /etc/slurm/power_save.conf
COPY usr/local/sbin/power_manager.sh /usr/local/sbin/power_manager.sh

RUN chmod +x /usr/local/sbin/power_manager.sh

WORKDIR /root