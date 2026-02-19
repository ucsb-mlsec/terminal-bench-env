FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages requests>=2.28

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/user/fileserver \
    && mkdir -p /home/user/solution \
    && mkdir -p /var/www/public/subdirectory \
    && mkdir -p /home/user

COPY fileserver/server.py /home/user/fileserver/server.py
COPY fileserver/test_security.py /home/user/fileserver/test_security.py
COPY data/public/index.html /var/www/public/index.html
COPY data/public/subdirectory/data.txt /var/www/public/subdirectory/data.txt
COPY data/secrets.txt /home/user/secrets.txt

RUN touch /etc/passwd && chmod 644 /etc/passwd

WORKDIR /home/user