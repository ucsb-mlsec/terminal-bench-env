FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/appserver/lib-v2.1.0 \
    /opt/appserver/config-v2.1.0 \
    /opt/appserver/bin-v2.1.0 \
    /opt/appserver/logs \
    /home/agent

COPY opt/appserver/lib /opt/appserver/lib
COPY opt/appserver/config /opt/appserver/config
COPY opt/appserver/bin /opt/appserver/bin
COPY opt/appserver/lib-v2.1.0/commons.jar /opt/appserver/lib-v2.1.0/commons.jar
COPY opt/appserver/config-v2.1.0/application.conf /opt/appserver/config-v2.1.0/application.conf
COPY opt/appserver/bin-v2.1.0/startup.sh /opt/appserver/bin-v2.1.0/startup.sh
COPY opt/appserver/deployment.conf /opt/appserver/deployment.conf
COPY opt/appserver/logs/deployment.log /opt/appserver/logs/deployment.log

RUN chmod +x /opt/appserver/bin-v2.1.0/startup.sh

WORKDIR /home/agent