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
    apt-get install -y stunnel4 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y openssl curl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/backend /etc/stunnel/certs /home/agent

COPY opt/backend/server.py /opt/backend/server.py
COPY etc/stunnel/certs/server.crt /etc/stunnel/certs/server.crt
COPY etc/stunnel/certs/server.key /etc/stunnel/certs/server.key
COPY etc/stunnel/stunnel.conf /etc/stunnel/stunnel.conf

RUN chmod 600 /etc/stunnel/certs/server.key && \
    chmod 644 /etc/stunnel/certs/server.crt && \
    chmod +x /opt/backend/server.py

WORKDIR /home/agent