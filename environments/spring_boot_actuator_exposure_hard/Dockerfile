FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/springapp/config /tmp

COPY springapp/application.properties /opt/springapp/application.properties
COPY springapp/pom.xml /opt/springapp/pom.xml
COPY springapp/config/application-prod.yml /opt/springapp/config/application-prod.yml
COPY springapp/config/bootstrap.properties /opt/springapp/config/bootstrap.properties

WORKDIR /opt/springapp