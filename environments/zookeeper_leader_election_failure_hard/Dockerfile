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
    openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/zookeeper \
    /var/lib/zookeeper/data \
    /var/log/zookeeper \
    /tmp

COPY config/zoo.cfg /etc/zookeeper/zoo.cfg
COPY config/zoo.cfg.template /etc/zookeeper/zoo.cfg.template
COPY config/zoo.cfg.backup /etc/zookeeper/zoo.cfg.backup
COPY config/zookeeper.properties /etc/zookeeper/zookeeper.properties
COPY config/myid.template /etc/zookeeper/myid.template
COPY config/log4j.properties /etc/zookeeper/log4j.properties
COPY config/java.env /etc/zookeeper/java.env
COPY config/dynamicConfigFile /etc/zookeeper/dynamicConfigFile
COPY data/zookeeper.log /var/log/zookeeper/zookeeper.log

WORKDIR /workspace