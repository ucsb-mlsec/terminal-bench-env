FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /forensics/images /solution

COPY data/forensics/manifest.txt /forensics/manifest.txt
COPY data/forensics/images/image_01.img /forensics/images/image_01.img
COPY data/forensics/images/image_02.img /forensics/images/image_02.img
COPY data/forensics/images/image_03.img /forensics/images/image_03.img
COPY data/forensics/images/image_04.img /forensics/images/image_04.img
COPY data/forensics/images/image_05.img /forensics/images/image_05.img
COPY data/forensics/images/image_06.img /forensics/images/image_06.img

WORKDIR /forensics