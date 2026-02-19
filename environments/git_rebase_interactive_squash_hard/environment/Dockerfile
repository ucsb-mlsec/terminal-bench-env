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
    apt-get install -y git jq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/messy-repo/.git/config.d \
    /workspace/messy-repo/src/config

COPY messy-repo/dot-git/config /workspace/messy-repo/.git/config
COPY messy-repo/README.md /workspace/messy-repo/README.md
COPY messy-repo/package.json /workspace/messy-repo/package.json
COPY messy-repo/.gitignore /workspace/messy-repo/.gitignore
COPY messy-repo/src/index.js /workspace/messy-repo/src/index.js
COPY messy-repo/src/config/database.js /workspace/messy-repo/src/config/database.js
COPY messy-repo-commits.json /workspace/messy-repo-commits.json

RUN git config --global user.email "cleanup@example.com" && \
    git config --global user.name "Git Cleanup Bot" && \
    git config --global init.defaultBranch main

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
