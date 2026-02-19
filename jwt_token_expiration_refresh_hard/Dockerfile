FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /app

COPY auth_api.py /app/auth_api.py
COPY tokens.py /app/tokens.py
COPY db.py /app/db.py
COPY schema.sql /app/schema.sql
COPY requirements.txt /app/requirements.txt
COPY test_security.py /app/test_security.py

RUN pip3 install --no-cache-dir --break-system-packages flask pyjwt pytest

WORKDIR /app