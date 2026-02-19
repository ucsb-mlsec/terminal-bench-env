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
    apt-get install -y openssl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/nginx_test/sites-enabled /root/nginx_test/certs

COPY nginx_test/sites-enabled/site1.conf /root/nginx_test/sites-enabled/site1.conf
COPY nginx_test/sites-enabled/site2.conf /root/nginx_test/sites-enabled/site2.conf
COPY nginx_test/sites-enabled/site3.conf /root/nginx_test/sites-enabled/site3.conf
COPY nginx_test/sites-enabled/site4.conf /root/nginx_test/sites-enabled/site4.conf
COPY nginx_test/sites-enabled/site5.conf /root/nginx_test/sites-enabled/site5.conf
COPY nginx_test/certs/api.example.com.crt /root/nginx_test/certs/api.example.com.crt
COPY nginx_test/certs/customer1.crt /root/nginx_test/certs/customer1.crt
COPY nginx_test/certs/shop.example.com.crt /root/nginx_test/certs/shop.example.com.crt
COPY nginx_test/certs/customer2-blog.crt /root/nginx_test/certs/customer2-blog.crt
COPY nginx_test/certs/acme-portal.crt /root/nginx_test/certs/acme-portal.crt

WORKDIR /root

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
