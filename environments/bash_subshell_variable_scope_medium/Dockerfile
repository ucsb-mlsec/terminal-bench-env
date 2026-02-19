FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/scripts /var/log/access /tmp

COPY opt/scripts/process_logs.sh /opt/scripts/process_logs.sh
COPY data/var/log/access/access1.log /var/log/access/access1.log
COPY data/var/log/access/access2.log /var/log/access/access2.log
COPY data/var/log/access/access3.log /var/log/access/access3.log

RUN chmod +x /opt/scripts/process_logs.sh

WORKDIR /opt/scripts