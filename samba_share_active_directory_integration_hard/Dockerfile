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
    samba \
    winbind \
    krb5-user \
    libpam-winbind \
    libnss-winbind && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/samba \
    /etc \
    /var/ldap \
    /srv/samba/shares/finance \
    /srv/samba/shares/engineering \
    /srv/samba/shares/public \
    /root

COPY config/smb.conf /etc/samba/smb.conf
COPY config/nsswitch.conf /etc/nsswitch.conf
COPY config/krb5.conf /etc/krb5.conf
COPY data/users.ldif /var/ldap/users.ldif
COPY data/groups.ldif /var/ldap/groups.ldif

WORKDIR /root