FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/config /workspace/logs

COPY target.c /workspace/target.c
COPY config/incomplete.properties /workspace/config/incomplete.properties
COPY config/reference.properties /workspace/config/reference.properties
COPY logs/previous_run.log /workspace/logs/previous_run.log
COPY notes.txt /workspace/notes.txt

WORKDIR /workspace