FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/games

COPY data/game_001.txt /workspace/games/game_001.txt
COPY data/game_002.txt /workspace/games/game_002.txt
COPY data/game_003.txt /workspace/games/game_003.txt
COPY data/game_004.txt /workspace/games/game_004.txt
COPY data/game_005.txt /workspace/games/game_005.txt
COPY data/game_006.txt /workspace/games/game_006.txt
COPY data/game_007.txt /workspace/games/game_007.txt
COPY data/game_008.txt /workspace/games/game_008.txt
COPY data/game_009.txt /workspace/games/game_009.txt
COPY data/game_010.txt /workspace/games/game_010.txt
COPY data/game_011.txt /workspace/games/game_011.txt
COPY data/game_012.txt /workspace/games/game_012.txt
COPY data/game_013.txt /workspace/games/game_013.txt
COPY data/game_014.txt /workspace/games/game_014.txt
COPY data/game_015.txt /workspace/games/game_015.txt

RUN pip3 install --no-cache-dir --break-system-packages python-chess

WORKDIR /workspace