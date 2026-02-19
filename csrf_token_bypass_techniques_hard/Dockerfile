FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/webapp /home/agent

RUN pip3 install --no-cache-dir --break-system-packages flask

COPY app.py /opt/webapp/app.py
COPY sample_request.txt /opt/webapp/sample_request.txt
COPY README.md /opt/webapp/README.md

WORKDIR /home/agent