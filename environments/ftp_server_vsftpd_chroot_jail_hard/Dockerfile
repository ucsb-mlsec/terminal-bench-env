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
    apt-get install -y vsftpd && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/vsftpd /home/ftpadmin /home/ftpuser1 /home/ftpuser2 /home/ftppower1 /home/ftpupload1/uploads /root

COPY config/vsftpd.conf /etc/vsftpd/vsftpd.conf
COPY data/user_access_policy.txt /home/ftpadmin/user_access_policy.txt

RUN useradd -d /home/ftpuser1 -s /bin/bash ftpuser1 && \
    useradd -d /home/ftpuser2 -s /bin/bash ftpuser2 && \
    useradd -d /home/ftppower1 -s /bin/bash ftppower1 && \
    useradd -d /home/ftpupload1 -s /bin/bash ftpupload1 && \
    chown ftpuser1:ftpuser1 /home/ftpuser1 && \
    chown ftpuser2:ftpuser2 /home/ftpuser2 && \
    chown ftppower1:ftppower1 /home/ftppower1 && \
    chown root:root /home/ftpupload1 && \
    chown ftpupload1:ftpupload1 /home/ftpupload1/uploads

WORKDIR /root