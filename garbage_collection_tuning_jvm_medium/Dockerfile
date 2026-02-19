FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /logs /configs /solution

COPY data/gc_pattern_1.log /logs/gc_pattern_1.log
COPY data/gc_pattern_2.log /logs/gc_pattern_2.log
COPY data/gc_pattern_3.log /logs/gc_pattern_3.log
COPY config/profile_a.txt /configs/profile_a.txt
COPY config/profile_b.txt /configs/profile_b.txt
COPY config/profile_c.txt /configs/profile_c.txt

WORKDIR /solution