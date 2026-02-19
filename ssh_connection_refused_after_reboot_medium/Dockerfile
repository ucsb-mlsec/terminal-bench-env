FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Install SSH server and necessary tools
RUN apt-get update && \
    apt-get install -y openssh-server systemctl && \
    rm -rf /var/lib/apt/lists/*

# Create necessary directories for snapshot files and SSH
RUN mkdir -p /var/log/system_snapshot /root /run/sshd /var/run/sshd

# Copy snapshot files to the expected location
COPY data/sshd_config /var/log/system_snapshot/sshd_config
COPY data/service_status.txt /var/log/system_snapshot/service_status.txt
COPY data/enabled_services.txt /var/log/system_snapshot/enabled_services.txt
COPY data/port_bindings.txt /var/log/system_snapshot/port_bindings.txt
COPY data/system.log /var/log/system_snapshot/system.log

WORKDIR /root