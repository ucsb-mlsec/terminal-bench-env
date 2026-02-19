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
    apt-get install -y --no-install-recommends \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages PyJWT cryptography

RUN mkdir -p /opt/api_service/keys /opt/api_service/test_tokens

COPY data/public_key.pem /opt/api_service/keys/public_key.pem
COPY data/valid_tokens.txt /opt/api_service/test_tokens/valid_tokens.txt
COPY data/invalid_tokens.txt /opt/api_service/test_tokens/invalid_tokens.txt

WORKDIR /opt/api_service