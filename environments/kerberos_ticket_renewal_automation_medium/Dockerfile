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
    krb5-user \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y krb5-user cron && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/kerberos /etc/security/keytabs

COPY config/svc_analytics.keytab /etc/security/keytabs/svc_analytics.keytab

RUN chmod 600 /etc/security/keytabs/svc_analytics.keytab

WORKDIR /opt/kerberos