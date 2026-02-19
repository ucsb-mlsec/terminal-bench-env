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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages protobuf

RUN pip3 install --no-cache-dir --break-system-packages grpcio grpcio-tools grpcio-reflection

RUN mkdir -p /workspace

COPY grpc_server.py /workspace/grpc_server.py
COPY calculator_pb2.py /workspace/calculator_pb2.py
COPY calculator_pb2_grpc.py /workspace/calculator_pb2_grpc.py
COPY user_pb2.py /workspace/user_pb2.py
COPY user_pb2_grpc.py /workspace/user_pb2_grpc.py

WORKDIR /workspace