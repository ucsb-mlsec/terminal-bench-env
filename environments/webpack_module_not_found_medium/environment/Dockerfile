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

RUN mkdir -p /workspace/project/src/components /workspace/project/src/utils /workspace/project/src/helpers

COPY project/package.json /workspace/project/package.json
COPY project/webpack.config.js /workspace/project/webpack.config.js
COPY project/src/index.js /workspace/project/src/index.js
COPY project/src/components/Button.js /workspace/project/src/components/Button.js
COPY project/src/utils/dateUtils.js /workspace/project/src/utils/dateUtils.js
COPY project/src/utils/domUtils.js /workspace/project/src/utils/domUtils.js
COPY project/src/helpers/validation.js /workspace/project/src/helpers/validation.js

WORKDIR /workspace/project

RUN npm install

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
