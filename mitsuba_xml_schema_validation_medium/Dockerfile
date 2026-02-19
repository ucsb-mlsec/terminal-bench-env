FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages lxml

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/scenes /workspace/docs

COPY data/scenes/scene1.xml /workspace/scenes/scene1.xml
COPY data/scenes/scene2.xml /workspace/scenes/scene2.xml
COPY data/scenes/scene3.xml /workspace/scenes/scene3.xml
COPY data/scenes/scene4.xml /workspace/scenes/scene4.xml
COPY data/scenes/scene5.xml /workspace/scenes/scene5.xml
COPY data/scenes/scene6.xml /workspace/scenes/scene6.xml
COPY data/scenes/scene7.xml /workspace/scenes/scene7.xml
COPY data/scenes/valid_complete.xml /workspace/scenes/valid_complete.xml
COPY data/docs/mitsuba_schema.txt /workspace/docs/mitsuba_schema.txt

WORKDIR /workspace