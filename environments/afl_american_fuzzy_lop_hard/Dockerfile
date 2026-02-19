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
    build-essential \
    clang \
    llvm \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc clang make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/benchmark/imgproc/samples

COPY imgproc/processor.c /home/benchmark/imgproc/processor.c
COPY imgproc/samples/test1.jpg /home/benchmark/imgproc/samples/test1.jpg
COPY imgproc/samples/test2.png /home/benchmark/imgproc/samples/test2.png
COPY imgproc/samples/test3.bmp /home/benchmark/imgproc/samples/test3.bmp

WORKDIR /home/benchmark