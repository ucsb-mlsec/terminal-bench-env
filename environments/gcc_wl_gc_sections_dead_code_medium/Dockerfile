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
    apt-get install -y build-essential gcc make binutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/bloated_project

COPY bloated_project/util1.c /workspace/bloated_project/util1.c
COPY bloated_project/util2.c /workspace/bloated_project/util2.c
COPY bloated_project/util3.c /workspace/bloated_project/util3.c
COPY bloated_project/util4.c /workspace/bloated_project/util4.c
COPY bloated_project/util5.c /workspace/bloated_project/util5.c
COPY bloated_project/util6.c /workspace/bloated_project/util6.c
COPY bloated_project/util7.c /workspace/bloated_project/util7.c
COPY bloated_project/util8.c /workspace/bloated_project/util8.c
COPY bloated_project/main.c /workspace/bloated_project/main.c
COPY bloated_project/Makefile /workspace/bloated_project/Makefile

WORKDIR /workspace