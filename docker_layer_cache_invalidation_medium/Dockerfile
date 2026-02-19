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

WORKDIR /workspace/nodeapp

COPY nodeapp/package.json /workspace/nodeapp/package.json
COPY nodeapp/package-lock.json /workspace/nodeapp/package-lock.json

RUN npm install

COPY nodeapp/server.js /workspace/nodeapp/server.js
COPY nodeapp/routes/index.js /workspace/nodeapp/routes/index.js
COPY nodeapp/public/index.html /workspace/nodeapp/public/index.html