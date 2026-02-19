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

RUN pip3 install --no-cache-dir --break-system-packages pyjwt cryptography

RUN mkdir -p /opt/auth-service/data /solution

COPY jwt_validator.py /opt/auth-service/jwt_validator.py
COPY data/public_key_1.pem /opt/auth-service/data/public_key_1.pem
COPY data/public_key_2.pem /opt/auth-service/data/public_key_2.pem
COPY data/token_key1.txt /opt/auth-service/data/token_key1.txt
COPY data/token_key2.txt /opt/auth-service/data/token_key2.txt
COPY run_tests.sh /opt/auth-service/run_tests.sh

RUN chmod +x /opt/auth-service/run_tests.sh

WORKDIR /opt/auth-service

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
