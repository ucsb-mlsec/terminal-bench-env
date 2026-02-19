FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/netns_configs /tmp

COPY config/service1.conf /opt/netns_configs/service1.conf
COPY config/service2.conf /opt/netns_configs/service2.conf
COPY config/service3.conf /opt/netns_configs/service3.conf
COPY config/service4.conf /opt/netns_configs/service4.conf
COPY config/service5.conf /opt/netns_configs/service5.conf
COPY config/service6.conf /opt/netns_configs/service6.conf
COPY config/service7.conf /opt/netns_configs/service7.conf

WORKDIR /workspace