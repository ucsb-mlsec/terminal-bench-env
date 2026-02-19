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
    apt-get install -y wget openjdk-17-jre-headless && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/project/logs /home/project

COPY data/api-requirements.txt /home/project/api-requirements.txt
COPY data/logs/payment-api.log /home/project/logs/payment-api.log
COPY data/logs/inventory-api.log /home/project/logs/inventory-api.log
COPY data/logs/shipping-api.log /home/project/logs/shipping-api.log

RUN cd /opt && \
    wget -q https://repo1.maven.org/maven2/org/wiremock/wiremock-standalone/3.3.1/wiremock-standalone-3.3.1.jar -O wiremock-standalone.jar

WORKDIR /home/project