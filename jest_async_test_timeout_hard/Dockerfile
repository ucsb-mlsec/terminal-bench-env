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
    apt-get install -y nodejs npm && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/payment-service

COPY payment-service/jest.config.js /workspace/payment-service/jest.config.js
COPY payment-service/package.json /workspace/payment-service/package.json
COPY payment-service/server.js /workspace/payment-service/server.js
COPY payment-service/database.js /workspace/payment-service/database.js
COPY payment-service/api.test.js /workspace/payment-service/api.test.js

WORKDIR /workspace/payment-service

RUN npm install

WORKDIR /workspace