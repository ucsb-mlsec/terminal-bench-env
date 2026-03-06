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
    apt-get install -y sqlite3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages flask

RUN mkdir -p /var/log/webapp /var/log/mail /var/backups /etc/nginx/sites-available /opt/webapp /root

COPY data/access.log /var/log/webapp/access.log
COPY data/error.log /var/log/webapp/error.log
COPY data/smtp.log /var/log/mail/smtp.log
COPY data/sessions.db /var/backups/sessions.db
COPY config/webapp.conf /etc/nginx/sites-available/webapp.conf
COPY app.py /opt/webapp/app.py
COPY utils.py /opt/webapp/utils.py
COPY auth.py /opt/webapp/auth.py

WORKDIR /root

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
