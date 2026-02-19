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
    apt-get install -y protobuf-compiler python3-protobuf && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages protobuf

RUN mkdir -p /challenge/tokens /solution

COPY challenge/schema.proto /challenge/schema.proto
COPY challenge/validator.py /challenge/validator.py
COPY challenge/tokens/user1.txt /challenge/tokens/user1.txt
COPY challenge/tokens/user2.txt /challenge/tokens/user2.txt
COPY challenge/tokens/user3.txt /challenge/tokens/user3.txt
COPY challenge/tokens/user4.txt /challenge/tokens/user4.txt

WORKDIR /workspace