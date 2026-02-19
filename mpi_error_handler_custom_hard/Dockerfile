FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace

COPY data/numbers.txt /workspace/numbers.txt
COPY data/progress_0.txt /workspace/progress_0.txt
COPY data/progress_1.txt /workspace/progress_1.txt
COPY data/progress_2.txt /workspace/progress_2.txt
COPY data/progress_3.txt /workspace/progress_3.txt

WORKDIR /workspace