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
    apt-get install -y build-essential g++ make time && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/project

COPY project/model1.cpp /workspace/project/model1.cpp
COPY project/model2.cpp /workspace/project/model2.cpp
COPY project/model3.cpp /workspace/project/model3.cpp
COPY project/Makefile /workspace/project/Makefile

WORKDIR /workspace