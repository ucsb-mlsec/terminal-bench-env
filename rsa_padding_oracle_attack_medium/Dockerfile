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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages cryptography

RUN pip3 install --no-cache-dir --break-system-packages pycryptodome

RUN mkdir -p /challenge /solution /tmp

COPY challenge/encrypted_message.bin /challenge/encrypted_message.bin
COPY challenge/public_key.pem /challenge/public_key.pem
COPY challenge/private_key.pem /challenge/private_key.pem
COPY challenge/oracle.py /challenge/oracle.py

RUN chmod +x /challenge/oracle.py && chmod 644 /challenge/*.bin /challenge/*.pem

WORKDIR /workspace