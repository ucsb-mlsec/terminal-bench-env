FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask requests

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/oauth_server /tmp

COPY opt/oauth_server/server.py /opt/oauth_server/server.py
COPY opt/oauth_server/test_client.py /opt/oauth_server/test_client.py
COPY opt/oauth_server/requirements.txt /opt/oauth_server/requirements.txt

RUN pip3 install --no-cache-dir --break-system-packages -r /opt/oauth_server/requirements.txt

WORKDIR /opt/oauth_server