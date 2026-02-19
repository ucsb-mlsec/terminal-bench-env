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

RUN mkdir -p /workspace/project/src /workspace/project/shared

COPY project/src/index.ts /workspace/project/src/index.ts
COPY project/src/app.ts /workspace/project/src/app.ts
COPY project/shared/utils.ts /workspace/project/shared/utils.ts
COPY project/shared/types.ts /workspace/project/shared/types.ts
COPY project/package.json /workspace/project/package.json

WORKDIR /workspace/project

RUN npm install