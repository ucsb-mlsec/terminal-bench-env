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
    apt-get install -y php-cli php-common && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/challenge/data \
    /home/challenge/files \
    /home/challenge/config \
    /tmp/php_sessions

COPY file_viewer.php /home/challenge/file_viewer.php
COPY data/requests.log /home/challenge/requests.log
COPY data/flag.txt /home/challenge/data/flag.txt
COPY files/document.txt /home/challenge/files/document.txt
COPY files/page.php /home/challenge/files/page.php
COPY config/database.php /home/challenge/config/database.php

RUN chmod 755 /home/challenge && \
    chmod 644 /home/challenge/file_viewer.php \
    /home/challenge/requests.log \
    /home/challenge/data/flag.txt \
    /home/challenge/files/document.txt \
    /home/challenge/files/page.php \
    /home/challenge/config/database.php && \
    chmod 777 /tmp/php_sessions

WORKDIR /home/challenge