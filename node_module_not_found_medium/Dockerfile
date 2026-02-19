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
    apt-get install -y curl nodejs npm && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/data /solution

COPY weather-service.js /app/weather-service.js
COPY package.json /app/package.json
COPY data/observations.json /app/data/observations.json

WORKDIR /app