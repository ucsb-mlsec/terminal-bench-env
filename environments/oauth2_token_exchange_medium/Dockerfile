FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/oauth2_mock /workspace

COPY opt/oauth2_mock/server.py /opt/oauth2_mock/server.py
COPY client_credentials.txt /workspace/client_credentials.txt
COPY auth_code.txt /workspace/auth_code.txt

RUN pip3 install --no-cache-dir --break-system-packages requests

WORKDIR /workspace