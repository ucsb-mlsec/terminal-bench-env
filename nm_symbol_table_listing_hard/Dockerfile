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
    apt-get install -y binutils && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/objects

COPY data/objects/module1.o /workspace/objects/module1.o
COPY data/objects/module2.o /workspace/objects/module2.o
COPY data/objects/module3.o /workspace/objects/module3.o
COPY data/objects/module4.o /workspace/objects/module4.o
COPY data/objects/module5.o /workspace/objects/module5.o
COPY data/objects/module6.o /workspace/objects/module6.o
COPY data/objects/module7.o /workspace/objects/module7.o
COPY data/objects/module8.o /workspace/objects/module8.o
COPY data/objects/module9.o /workspace/objects/module9.o
COPY data/objects/module10.o /workspace/objects/module10.o

WORKDIR /workspace