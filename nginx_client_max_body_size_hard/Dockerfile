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
    apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask werkzeug

RUN mkdir -p /config/nginx/conf.d \
    /config/nginx/sites-available \
    /data \
    /tmp

COPY config/nginx/nginx.conf /config/nginx/nginx.conf
COPY config/nginx/conf.d/legacy.conf /config/nginx/conf.d/legacy.conf
COPY config/nginx/backend1.conf /config/nginx/backend1.conf
COPY config/nginx/backend2.conf /config/nginx/backend2.conf
COPY config/nginx/backend3.conf /config/nginx/backend3.conf
COPY config/nginx/sites-available/additional.conf /config/nginx/sites-available/additional.conf
COPY data/nginx_error.log /data/nginx_error.log
COPY data/nginx_access.log /data/nginx_access.log
COPY app.py /app.py

WORKDIR /tmp