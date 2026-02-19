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
    apt-get install -y \
    apache2 \
    php \
    libapache2-mod-php \
    selinux-utils \
    policycoreutils \
    selinux-policy-default \
    auditd && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y policycoreutils-python-utils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/webapp/html && \
    mkdir -p /etc/httpd/conf && \
    mkdir -p /var/log/audit

COPY data/index.html /opt/webapp/html/index.html
COPY data/about.php /opt/webapp/html/about.php
COPY data/contact.php /opt/webapp/html/contact.php
COPY data/services.php /opt/webapp/html/services.php
COPY config/httpd.conf /etc/httpd/conf/httpd.conf
COPY data/audit.log /var/log/audit/audit.log

RUN ln -sf /etc/httpd/conf/httpd.conf /etc/apache2/apache2.conf || true

WORKDIR /root