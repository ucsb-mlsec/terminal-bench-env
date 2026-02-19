FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /etc/netcap /usr/local/bin /home/user

COPY config/config.json /etc/netcap/config.json
COPY config/interfaces.conf /etc/netcap/interfaces.conf
COPY config/permissions.conf /etc/netcap/permissions.conf
COPY config/system_limits.conf /etc/netcap/system_limits.conf
COPY config/environment.conf /etc/netcap/environment.conf
COPY config/interface_priorities.conf /etc/netcap/interface_priorities.conf
COPY usr/local/bin/mock_tcpdump /usr/local/bin/mock_tcpdump

RUN chmod +x /usr/local/bin/mock_tcpdump

WORKDIR /home/user