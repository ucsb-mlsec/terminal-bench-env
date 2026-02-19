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
    curl \
    openjdk-11-jre-headless \
    nodejs \
    npm && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y default-jre-headless curl && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g google-closure-compiler

RUN mkdir -p /workspace/src /workspace/test /workspace/dist

COPY src/utils.js /workspace/src/utils.js
COPY test/validate.js /workspace/test/validate.js

WORKDIR /workspace