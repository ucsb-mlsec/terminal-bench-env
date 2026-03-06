FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/challenge/positions /home/challenge

COPY data/board1.json /home/challenge/positions/board1.json
COPY data/board2.json /home/challenge/positions/board2.json
COPY data/board3.json /home/challenge/positions/board3.json
COPY data/board4.json /home/challenge/positions/board4.json
COPY data/board5.json /home/challenge/positions/board5.json

WORKDIR /home/challenge

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
