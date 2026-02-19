FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/positions

COPY data/positions/board_1.txt /workspace/positions/board_1.txt
COPY data/positions/board_2.txt /workspace/positions/board_2.txt
COPY data/positions/board_3.txt /workspace/positions/board_3.txt
COPY data/positions/board_4.txt /workspace/positions/board_4.txt
COPY data/positions/board_5.txt /workspace/positions/board_5.txt
COPY data/positions/board_6.txt /workspace/positions/board_6.txt
COPY data/positions/board_7.txt /workspace/positions/board_7.txt
COPY data/positions/board_8.txt /workspace/positions/board_8.txt

WORKDIR /workspace