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

RUN mkdir -p /workspace/exports

COPY user_profile.proto /workspace/user_profile.proto
COPY user_profile_original.proto /workspace/user_profile_original.proto
COPY data/export1.txtpb /workspace/exports/export1.txtpb
COPY data/export2.txtpb /workspace/exports/export2.txtpb
COPY data/export3.txtpb /workspace/exports/export3.txtpb
COPY parse_test.sh /workspace/parse_test.sh

RUN chmod +x /workspace/parse_test.sh

WORKDIR /workspace