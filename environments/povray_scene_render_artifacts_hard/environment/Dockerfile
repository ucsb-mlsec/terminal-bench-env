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

COPY data/scenes/scene1.pov /workspace/scenes/scene1.pov
COPY data/scenes/scene2.pov /workspace/scenes/scene2.pov
COPY data/scenes/scene3.pov /workspace/scenes/scene3.pov
COPY data/scenes/scene4.pov /workspace/scenes/scene4.pov
COPY data/scenes/scene5.pov /workspace/scenes/scene5.pov
COPY data/scenes/scene6.pov /workspace/scenes/scene6.pov

WORKDIR /workspace

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
