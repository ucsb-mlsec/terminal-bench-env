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
    apt-get install -y file && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y grep && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/scripts \
    /workspace/deploy_repo/config \
    /workspace/deploy_repo/docs \
    /workspace/deploy_repo/build \
    /workspace/deploy_repo/lib \
    /workspace/deploy_repo/assets \
    /workspace/deploy_repo/data \
    /workspace/deploy_repo/archive \
    /workspace/deploy_repo/bin \
    /workspace/deploy_repo/logs \
    /workspace/deploy_repo/scripts \
    /workspace/deploy_repo/monitoring \
    /solution

COPY scripts/search_config.sh /workspace/scripts/search_config.sh
COPY data/deploy_repo/config/api.conf /workspace/deploy_repo/config/api.conf
COPY data/deploy_repo/config/services.yaml /workspace/deploy_repo/config/services.yaml
COPY data/deploy_repo/config/database.json /workspace/deploy_repo/config/database.json
COPY data/deploy_repo/docs/README.txt /workspace/deploy_repo/docs/README.txt
COPY data/deploy_repo/app.conf /workspace/deploy_repo/app.conf
COPY data/deploy_repo/build/app.o /workspace/deploy_repo/build/app.o
COPY data/deploy_repo/lib/libutil.so /workspace/deploy_repo/lib/libutil.so
COPY data/deploy_repo/assets/logo.png /workspace/deploy_repo/assets/logo.png
COPY data/deploy_repo/data/cache.db /workspace/deploy_repo/data/cache.db
COPY data/deploy_repo/archive/backup.tar.gz /workspace/deploy_repo/archive/backup.tar.gz
COPY data/deploy_repo/bin/deploy /workspace/deploy_repo/bin/deploy
COPY data/deploy_repo/logs/deployment.log /workspace/deploy_repo/logs/deployment.log
COPY data/deploy_repo/scripts/deploy.sh /workspace/deploy_repo/scripts/deploy.sh
COPY data/deploy_repo/monitoring/metrics.json /workspace/deploy_repo/monitoring/metrics.json

RUN chmod +x /workspace/scripts/search_config.sh

WORKDIR /workspace