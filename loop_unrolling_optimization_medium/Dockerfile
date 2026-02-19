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
    apt-get install -y build-essential gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/imageproc/blur_filter

COPY blur_filter.c /home/imageproc/blur_filter/blur_filter.c
COPY Makefile /home/imageproc/blur_filter/Makefile
COPY data/input.pgm /home/imageproc/blur_filter/input.pgm

WORKDIR /home/imageproc/blur_filter