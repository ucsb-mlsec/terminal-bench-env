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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages flask gunicorn pandas

RUN mkdir -p /opt/webapp /opt/webapp/config /opt/webapp/data

COPY app.py /opt/webapp/app.py
COPY config/gunicorn_config.py /opt/webapp/gunicorn_config.py
COPY data/app.log /opt/webapp/app.log

WORKDIR /opt/webapp