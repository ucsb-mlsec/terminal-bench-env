FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN pip3 install --no-cache-dir --break-system-packages flask psutil

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/webapp/handlers /opt/webapp/utils /opt/webapp/data /tmp

COPY data/memory_snapshot.json /opt/webapp/data/memory_snapshot.json
COPY data/application.log /opt/webapp/data/application.log
COPY app.py /opt/webapp/app.py
COPY handlers/__init__.py /opt/webapp/handlers/__init__.py
COPY handlers/api.py /opt/webapp/handlers/api.py
COPY handlers/metrics.py /opt/webapp/handlers/metrics.py
COPY utils/helpers.py /opt/webapp/utils/helpers.py

WORKDIR /opt/webapp