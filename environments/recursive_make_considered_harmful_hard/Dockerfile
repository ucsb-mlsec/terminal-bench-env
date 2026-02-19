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
    apt-get install -y build-essential make g++ && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /project/common /project/network /project/app /tmp

COPY project/common/utils.h /project/common/utils.h
COPY project/common/utils.cpp /project/common/utils.cpp
COPY project/common/Makefile /project/common/Makefile
COPY project/network/socket.h /project/network/socket.h
COPY project/network/socket.cpp /project/network/socket.cpp
COPY project/network/Makefile /project/network/Makefile
COPY project/app/main.cpp /project/app/main.cpp
COPY project/app/Makefile /project/app/Makefile
COPY project/Makefile /project/Makefile

WORKDIR /project