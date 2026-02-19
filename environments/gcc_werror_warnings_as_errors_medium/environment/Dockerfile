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
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make libc6-dev && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /project/legacy_app

COPY project/legacy_app/main.c /project/legacy_app/main.c
COPY project/legacy_app/utils.c /project/legacy_app/utils.c
COPY project/legacy_app/utils.h /project/legacy_app/utils.h
COPY project/legacy_app/data_processor.c /project/legacy_app/data_processor.c
COPY project/legacy_app/Makefile /project/legacy_app/Makefile
COPY project/legacy_app/test_suite.sh /project/legacy_app/test_suite.sh

RUN chmod +x /project/legacy_app/test_suite.sh

WORKDIR /project/legacy_app

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
