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
    apt-get install -y --no-install-recommends \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages redis>=4.0

RUN pip3 install --no-cache-dir --break-system-packages celery

RUN mkdir -p /opt/celery_app/data

COPY celeryconfig.py /opt/celery_app/celeryconfig.py
COPY tasks.py /opt/celery_app/tasks.py
COPY process_data.py /opt/celery_app/process_data.py
COPY data/small.dat /opt/celery_app/data/small.dat
COPY data/medium.dat /opt/celery_app/data/medium.dat
COPY data/large.dat /opt/celery_app/data/large.dat

WORKDIR /opt/celery_app