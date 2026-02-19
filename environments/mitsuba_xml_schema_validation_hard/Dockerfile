FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/scenes

COPY data/validation_rules.txt /workspace/validation_rules.txt
COPY data/scenes/scene_01.xml /workspace/scenes/scene_01.xml
COPY data/scenes/scene_02.xml /workspace/scenes/scene_02.xml
COPY data/scenes/scene_03.xml /workspace/scenes/scene_03.xml
COPY data/scenes/scene_broken_01.xml /workspace/scenes/scene_broken_01.xml
COPY data/scenes/scene_broken_02.xml /workspace/scenes/scene_broken_02.xml
COPY data/scenes/scene_broken_03.xml /workspace/scenes/scene_broken_03.xml
COPY data/scenes/scene_broken_04.xml /workspace/scenes/scene_broken_04.xml
COPY data/scenes/scene_04.xml /workspace/scenes/scene_04.xml

WORKDIR /workspace