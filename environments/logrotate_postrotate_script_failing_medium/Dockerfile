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
    apt-get install -y logrotate && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/logrotate.d /var/log/webapp /tmp

COPY config/webapp /etc/logrotate.d/webapp
COPY data/access.log /var/log/webapp/access.log
COPY data/error.log /var/log/webapp/error.log

WORKDIR /workspace