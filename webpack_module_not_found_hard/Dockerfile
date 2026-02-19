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

RUN mkdir -p /workspace/monorepo/packages/utils/src && \
    mkdir -p /workspace/monorepo/packages/components/src && \
    mkdir -p /workspace/monorepo/packages/app/src && \
    mkdir -p /workspace/monorepo/packages/app/dist

COPY workspace/monorepo/package.json /workspace/monorepo/package.json
COPY workspace/monorepo/packages/utils/package.json /workspace/monorepo/packages/utils/package.json
COPY workspace/monorepo/packages/utils/src/index.js /workspace/monorepo/packages/utils/src/index.js
COPY workspace/monorepo/packages/components/package.json /workspace/monorepo/packages/components/package.json
COPY workspace/monorepo/packages/components/src/index.js /workspace/monorepo/packages/components/src/index.js
COPY workspace/monorepo/packages/app/package.json /workspace/monorepo/packages/app/package.json
COPY workspace/monorepo/packages/app/src/index.js /workspace/monorepo/packages/app/src/index.js
COPY workspace/monorepo/packages/app/webpack.config.js /workspace/monorepo/packages/app/webpack.config.js
COPY workspace/monorepo/packages/app/.babelrc /workspace/monorepo/packages/app/.babelrc

WORKDIR /workspace/monorepo

RUN npm install

WORKDIR /workspace/monorepo/packages/app