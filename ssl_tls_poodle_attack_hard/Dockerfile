FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/infrastructure/apache/sites \
    /opt/infrastructure/nginx \
    /opt/infrastructure/proxy \
    /opt/infrastructure/backend \
    /opt/infrastructure/web_servers \
    /opt/infrastructure/app_configs

COPY config/apache_secure.conf /opt/infrastructure/apache_secure.conf
COPY config/apache_vulnerable_sslv3.conf /opt/infrastructure/apache_vulnerable_sslv3.conf
COPY config/nginx_secure.conf /opt/infrastructure/nginx_secure.conf
COPY config/nginx_vulnerable_cbc.conf /opt/infrastructure/nginx_vulnerable_cbc.conf
COPY config/haproxy_secure.cfg /opt/infrastructure/haproxy_secure.cfg
COPY config/haproxy_vulnerable_sslv3.cfg /opt/infrastructure/haproxy_vulnerable_sslv3.cfg
COPY config/lighttpd_vulnerable.conf /opt/infrastructure/lighttpd_vulnerable.conf
COPY config/tomcat_server.xml /opt/infrastructure/tomcat_server.xml
COPY config/iis_bindings.config /opt/infrastructure/iis_bindings.config
COPY config/apache_mixed.conf /opt/infrastructure/apache_mixed.conf
COPY config/nginx_missing_protocol.conf /opt/infrastructure/nginx_missing_protocol.conf
COPY config/varnish_ssl.vcl /opt/infrastructure/varnish_ssl.vcl
COPY config/caddy_secure.conf /opt/infrastructure/caddy_secure.conf
COPY config/backend/api_service.conf /opt/infrastructure/backend/api_service.conf
COPY config/proxy/load_balancer.conf /opt/infrastructure/proxy/load_balancer.conf
COPY config/web_servers/legacy_site.conf /opt/infrastructure/web_servers/legacy_site.conf
COPY config/monitoring.conf /opt/infrastructure/monitoring.conf
COPY config/app_configs/microservice.yml /opt/infrastructure/app_configs/microservice.yml

WORKDIR /opt/infrastructure