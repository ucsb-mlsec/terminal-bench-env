FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /app/config /tmp

COPY config/application.properties /app/config/application.properties
COPY config/environment.yml /app/config/environment.yml
COPY config/services.json /app/config/services.json
COPY config/.env.backup /app/config/.env.backup
COPY config/settings.xml /app/config/settings.xml
COPY config/management.ini /app/config/management.ini

WORKDIR /app