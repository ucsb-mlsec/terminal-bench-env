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
    nodejs \
    npm && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/app/routes /workspace/logs /solution

COPY app/server.js /workspace/app/server.js
COPY app/database.js /workspace/app/database.js
COPY app/routes/users.js /workspace/app/routes/users.js
COPY data/error.log /workspace/logs/error.log

WORKDIR /workspace