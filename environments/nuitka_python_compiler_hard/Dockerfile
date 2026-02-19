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
    python3-pip \
    binutils \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages numpy pyinstaller

RUN mkdir -p /workspace/app /workspace/build_attempts /workspace/dist

COPY app/calculator.py /workspace/app/calculator.py
COPY build_attempts/attempt1.log /workspace/build_attempts/attempt1.log
COPY build_attempts/attempt2.log /workspace/build_attempts/attempt2.log
COPY build_attempts/notes.txt /workspace/build_attempts/notes.txt

WORKDIR /workspace