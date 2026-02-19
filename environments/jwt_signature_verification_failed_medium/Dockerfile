FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN pip3 install --no-cache-dir --break-system-packages pyjwt

RUN mkdir -p /app /app/secrets /app/test_tokens

COPY verify_service.py /app/verify_service.py
COPY data/current_secret.txt /app/secrets/current_secret.txt
COPY data/old_secret.txt /app/secrets/old_secret.txt
COPY data/token1.txt /app/test_tokens/token1.txt
COPY data/token2.txt /app/test_tokens/token2.txt
COPY data/token3.txt /app/test_tokens/token3.txt
COPY data/token4.txt /app/test_tokens/token4.txt
COPY data/token5.txt /app/test_tokens/token5.txt
COPY data/token6.txt /app/test_tokens/token6.txt

WORKDIR /app