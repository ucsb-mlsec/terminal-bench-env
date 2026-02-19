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
    apt-get install -y nginx openssl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/nginx/ssl /etc/nginx/certs /etc/nginx/wrong_location

COPY config/nginx/nginx.conf /etc/nginx/nginx.conf
COPY config/nginx/ssl/example.com.crt /etc/nginx/ssl/example.com.crt
COPY config/nginx/ssl/example.com.key /etc/nginx/ssl/example.com.key
COPY config/nginx/certs/testsite.com.crt /etc/nginx/certs/testsite.com.crt
COPY config/nginx/wrong_location/testsite.key /etc/nginx/wrong_location/testsite.key

WORKDIR /root