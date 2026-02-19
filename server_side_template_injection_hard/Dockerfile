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
RUN pip3 install --no-cache-dir --break-system-packages requests

RUN pip3 install --no-cache-dir --break-system-packages flask

RUN mkdir -p /opt/feedback_portal /opt/feedback_portal/data /tmp

COPY app.py /opt/feedback_portal/app.py
COPY data/exploits.txt /opt/feedback_portal/exploits.txt
COPY test_legitimate.py /opt/feedback_portal/test_legitimate.py

WORKDIR /opt/feedback_portal