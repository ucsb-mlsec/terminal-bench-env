FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/patterns

COPY patterns/pattern_01.txt /workspace/patterns/pattern_01.txt
COPY patterns/pattern_02.txt /workspace/patterns/pattern_02.txt
COPY patterns/pattern_03.txt /workspace/patterns/pattern_03.txt
COPY patterns/pattern_04.txt /workspace/patterns/pattern_04.txt
COPY patterns/pattern_05.txt /workspace/patterns/pattern_05.txt
COPY patterns/pattern_06.txt /workspace/patterns/pattern_06.txt
COPY patterns/pattern_07.txt /workspace/patterns/pattern_07.txt
COPY patterns/pattern_08.txt /workspace/patterns/pattern_08.txt

WORKDIR /workspace