FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /var/webapp/config \
    /var/webapp/static \
    /var/webapp/lib \
    /var/webapp/templates \
    /var/webapp/data \
    /var/webapp/resources/images \
    /var/webapp/docs \
    /tmp

COPY var/webapp/index.html /var/webapp/index.html
COPY var/webapp/config/app.conf /var/webapp/config/app.conf
COPY var/webapp/static/style.css /var/webapp/static/style.css
COPY var/webapp/lib/utils.js /var/webapp/lib/utils.js
COPY var/webapp/templates/layout.html /var/webapp/templates/layout.html
COPY var/webapp/data/users.json /var/webapp/data/users.json
COPY var/webapp/resources/images/logo.txt /var/webapp/resources/images/logo.txt
COPY var/webapp/docs/readme.txt /var/webapp/docs/readme.txt

WORKDIR /var/webapp