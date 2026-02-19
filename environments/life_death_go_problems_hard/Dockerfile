FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/boards /home/agent

COPY data/boards/board_01.txt /home/agent/boards/board_01.txt
COPY data/boards/board_02.txt /home/agent/boards/board_02.txt
COPY data/boards/board_03.txt /home/agent/boards/board_03.txt
COPY data/boards/board_04.txt /home/agent/boards/board_04.txt
COPY data/boards/board_05.txt /home/agent/boards/board_05.txt
COPY data/boards/board_06.txt /home/agent/boards/board_06.txt
COPY data/boards/board_07.txt /home/agent/boards/board_07.txt
COPY data/boards/board_08.txt /home/agent/boards/board_08.txt
COPY data/boards/board_09.txt /home/agent/boards/board_09.txt
COPY data/boards/board_10.txt /home/agent/boards/board_10.txt
COPY data/boards/board_11.txt /home/agent/boards/board_11.txt
COPY data/boards/board_12.txt /home/agent/boards/board_12.txt
COPY data/boards/board_13.txt /home/agent/boards/board_13.txt
COPY data/boards/board_14.txt /home/agent/boards/board_14.txt
COPY data/boards/board_15.txt /home/agent/boards/board_15.txt

WORKDIR /home/agent