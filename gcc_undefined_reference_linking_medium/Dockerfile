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
    apt-get install -y \
    build-essential \
    gcc \
    make \
    libsqlite3-dev \
    sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/sqlite_project

COPY sqlite_project/main.c /workspace/sqlite_project/main.c
COPY sqlite_project/db_utils.h /workspace/sqlite_project/db_utils.h
COPY sqlite_project/db_utils.c /workspace/sqlite_project/db_utils.c
COPY sqlite_project/Makefile /workspace/sqlite_project/Makefile

WORKDIR /workspace/sqlite_project