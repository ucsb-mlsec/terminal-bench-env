FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/ios_binaries

COPY data/ios_binaries/Instagram /workspace/ios_binaries/Instagram
COPY data/ios_binaries/Facebook /workspace/ios_binaries/Facebook
COPY data/ios_binaries/WhatsApp /workspace/ios_binaries/WhatsApp
COPY data/ios_binaries/Twitter /workspace/ios_binaries/Twitter
COPY data/ios_binaries/TikTok /workspace/ios_binaries/TikTok
COPY data/ios_binaries/Snapchat /workspace/ios_binaries/Snapchat
COPY data/ios_binaries/YouTube /workspace/ios_binaries/YouTube

WORKDIR /workspace