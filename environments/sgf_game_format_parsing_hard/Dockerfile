FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/sgf_files /workspace/solution

COPY data/sgf_files/simple_game.sgf /workspace/sgf_files/simple_game.sgf
COPY data/sgf_files/game_with_variations.sgf /workspace/sgf_files/game_with_variations.sgf
COPY data/sgf_files/deeply_nested.sgf /workspace/sgf_files/deeply_nested.sgf
COPY data/sgf_files/with_comments.sgf /workspace/sgf_files/with_comments.sgf
COPY data/sgf_files/with_setup.sgf /workspace/sgf_files/with_setup.sgf
COPY data/sgf_files/malformed_game.sgf /workspace/sgf_files/malformed_game.sgf
COPY data/sgf_files/complex_tree.sgf /workspace/sgf_files/complex_tree.sgf
COPY data/sgf_files/medium_game.sgf /workspace/sgf_files/medium_game.sgf

WORKDIR /workspace