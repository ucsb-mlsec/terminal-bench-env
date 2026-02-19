FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/chat_service

COPY chat_service/chat.proto /workspace/chat_service/chat.proto
COPY chat_service/server.py /workspace/chat_service/server.py
COPY chat_service/client.py /workspace/chat_service/client.py
COPY chat_service/requirements.txt /workspace/chat_service/requirements.txt
COPY chat_service/chat_pb2_grpc.py /workspace/chat_service/chat_pb2_grpc.py

RUN pip3 install --no-cache-dir --break-system-packages grpcio grpcio-tools

WORKDIR /workspace/chat_service