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

RUN mkdir -p /opt/nginx/configs /tmp

COPY config/service1.conf /opt/nginx/configs/service1.conf
COPY config/service2.conf /opt/nginx/configs/service2.conf
COPY config/service3.conf /opt/nginx/configs/service3.conf
COPY config/service4.conf /opt/nginx/configs/service4.conf
COPY config/service5.conf /opt/nginx/configs/service5.conf
COPY config/service6.conf /opt/nginx/configs/service6.conf
COPY config/service7.conf /opt/nginx/configs/service7.conf
COPY config/service8.conf /opt/nginx/configs/service8.conf
COPY config/service9.conf /opt/nginx/configs/service9.conf
COPY config/service10.conf /opt/nginx/configs/service10.conf

WORKDIR /workspace