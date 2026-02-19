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
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages requests

RUN pip3 install --no-cache-dir --break-system-packages flask pillow

RUN mkdir -p /opt/video_service/uploads /opt/video_service/test_data

COPY server.py /opt/video_service/server.py
COPY client_test.py /opt/video_service/client_test.py
COPY requirements.txt /opt/video_service/requirements.txt
COPY data/sample.mp4 /opt/video_service/test_data/sample.mp4

WORKDIR /opt/video_service