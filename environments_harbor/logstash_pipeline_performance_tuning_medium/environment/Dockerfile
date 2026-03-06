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
    wget \
    gnupg \
    apt-transport-https \
    openjdk-11-jdk && \
    wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add - && \
    echo "deb https://artifacts.elastic.co/packages/8.x/apt stable main" | tee -a /etc/apt/sources.list.d/elastic-8.x.list && \
    apt-get update && \
    apt-get install -y logstash && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/logstash/conf.d /home/agent /var/log/apache

COPY config/apache-pipeline.conf /etc/logstash/conf.d/apache-pipeline.conf

WORKDIR /home/agent

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
