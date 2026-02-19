FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/rotation /opt/rotation/test_data /solution

COPY handler.py /opt/rotation/handler.py
COPY db_mock.py /opt/rotation/db_mock.py
COPY secrets_mock.py /opt/rotation/secrets_mock.py
COPY run_tests.sh /opt/rotation/run_tests.sh

RUN chmod +x /opt/rotation/run_tests.sh

WORKDIR /workspace