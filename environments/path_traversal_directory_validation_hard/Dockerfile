FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/agent/docservice/tests /home/agent/docservice/documents

RUN pip3 install --no-cache-dir --break-system-packages flask

COPY docservice/app.py /home/agent/docservice/app.py
COPY docservice/tests/security_tests.py /home/agent/docservice/tests/security_tests.py
COPY docservice/documents/sample1.txt /home/agent/docservice/documents/sample1.txt
COPY docservice/documents/sample2.txt /home/agent/docservice/documents/sample2.txt

WORKDIR /home/agent/docservice