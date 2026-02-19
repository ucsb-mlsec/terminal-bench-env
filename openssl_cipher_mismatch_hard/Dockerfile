FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/ssl_configs/service_alpha \
    /opt/ssl_configs/service_beta \
    /opt/ssl_configs/service_gamma \
    /home/agent

COPY config/service_alpha_openssl.cnf /opt/ssl_configs/service_alpha/openssl.cnf
COPY config/service_beta_openssl.cnf /opt/ssl_configs/service_beta/openssl.cnf
COPY config/service_gamma_openssl.cnf /opt/ssl_configs/service_gamma/openssl.cnf

WORKDIR /home/agent