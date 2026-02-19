FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cpp_project/libcore \
    /workspace/cpp_project/libutils \
    /workspace/cpp_project/libnetwork \
    /workspace/cpp_project/libdata \
    /workspace/cpp_project/libui \
    /workspace/cpp_project/app

COPY cpp_project/libcore/Makefile /workspace/cpp_project/libcore/Makefile
COPY cpp_project/libutils/Makefile /workspace/cpp_project/libutils/Makefile
COPY cpp_project/libnetwork/Makefile /workspace/cpp_project/libnetwork/Makefile
COPY cpp_project/libdata/Makefile /workspace/cpp_project/libdata/Makefile
COPY cpp_project/libui/Makefile /workspace/cpp_project/libui/Makefile
COPY cpp_project/app/Makefile /workspace/cpp_project/app/Makefile

WORKDIR /workspace