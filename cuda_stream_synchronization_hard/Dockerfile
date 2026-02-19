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
    gcc \
    make && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y gcc make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/image_processor/test_images

COPY image_processor/image_processor.c /workspace/image_processor/image_processor.c
COPY image_processor/image_processor.h /workspace/image_processor/image_processor.h
COPY image_processor/Makefile /workspace/image_processor/Makefile
COPY image_processor/test_images/image1.pgm /workspace/image_processor/test_images/image1.pgm
COPY image_processor/test_images/image2.pgm /workspace/image_processor/test_images/image2.pgm
COPY image_processor/test_images/image3.pgm /workspace/image_processor/test_images/image3.pgm
COPY image_processor/test_images/image4.pgm /workspace/image_processor/test_images/image4.pgm
COPY image_processor/test_images/image5.pgm /workspace/image_processor/test_images/image5.pgm
COPY image_processor/run_tests.sh /workspace/image_processor/run_tests.sh

RUN chmod +x /workspace/image_processor/run_tests.sh

WORKDIR /workspace