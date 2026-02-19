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
    apt-get install -y grep file && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/project/src \
             /workspace/project/build \
             /workspace/project/assets \
             /workspace/project/lib \
             /workspace/project/dist \
             /workspace/project/data \
             /workspace/project/docs \
             /workspace/project/scripts \
             /workspace/project/config

COPY project/src/main.c /workspace/project/src/main.c
COPY project/src/utils.py /workspace/project/src/utils.py
COPY project/README.md /workspace/project/README.md
COPY project/build/app.o /workspace/project/build/app.o
COPY project/assets/logo.png /workspace/project/assets/logo.png
COPY project/assets/icon.jpg /workspace/project/assets/icon.jpg
COPY project/lib/libmath.a /workspace/project/lib/libmath.a
COPY project/dist/app.tar.gz /workspace/project/dist/app.tar.gz
COPY project/data/cache.dat /workspace/project/data/cache.dat
COPY project/docs/guide.txt /workspace/project/docs/guide.txt
COPY project/scripts/deploy.sh /workspace/project/scripts/deploy.sh
COPY project/config/settings.json /workspace/project/config/settings.json

WORKDIR /workspace