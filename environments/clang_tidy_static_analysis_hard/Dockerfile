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
    clang \
    clang-tidy \
    build-essential \
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y clang clang-tools build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/project/legacy-app/src /home/project/legacy-app/include

COPY legacy-app/src/main.cpp /home/project/legacy-app/src/main.cpp
COPY legacy-app/src/database.cpp /home/project/legacy-app/src/database.cpp
COPY legacy-app/src/utils.cpp /home/project/legacy-app/src/utils.cpp
COPY legacy-app/src/network.cpp /home/project/legacy-app/src/network.cpp
COPY legacy-app/src/processor.cpp /home/project/legacy-app/src/processor.cpp
COPY legacy-app/include/config.h /home/project/legacy-app/include/config.h
COPY legacy-app/include/database.h /home/project/legacy-app/include/database.h
COPY legacy-app/include/utils.h /home/project/legacy-app/include/utils.h
COPY legacy-app/include/network.h /home/project/legacy-app/include/network.h
COPY legacy-app/include/processor.h /home/project/legacy-app/include/processor.h
COPY legacy-app/Makefile /home/project/legacy-app/Makefile

WORKDIR /home/project/legacy-app