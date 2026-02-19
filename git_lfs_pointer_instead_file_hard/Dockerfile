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
    apt-get install -y git git-lfs && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/project-assets/images \
             /workspace/project-assets/data \
             /workspace/project-assets/assets \
             /workspace/project-assets/videos \
             /workspace/project-assets/docs \
             /workspace/project-assets/audio

COPY workspace/project-assets/images/logo.png /workspace/project-assets/images/logo.png
COPY workspace/project-assets/images/banner.jpg /workspace/project-assets/images/banner.jpg
COPY workspace/project-assets/data/model.bin /workspace/project-assets/data/model.bin
COPY workspace/project-assets/assets/font.ttf /workspace/project-assets/assets/font.ttf
COPY workspace/project-assets/videos/intro.mp4 /workspace/project-assets/videos/intro.mp4
COPY workspace/project-assets/images/icon.svg /workspace/project-assets/images/icon.svg
COPY workspace/project-assets/docs/readme.txt /workspace/project-assets/docs/readme.txt
COPY workspace/project-assets/audio/sound.wav /workspace/project-assets/audio/sound.wav
COPY workspace/project-assets/data/dataset.parquet /workspace/project-assets/data/dataset.parquet

WORKDIR /workspace