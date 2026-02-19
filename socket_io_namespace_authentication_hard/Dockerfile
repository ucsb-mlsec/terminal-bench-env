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

RUN mkdir -p /workspace/server /workspace/test-suite

COPY server/package.json /workspace/server/package.json
COPY server/server.js /workspace/server/server.js
COPY test-suite/verify-auth.js /workspace/test-suite/verify-auth.js

RUN cd /workspace/server && npm install

ENV JWT_SECRET=test-secret-key

WORKDIR /workspace