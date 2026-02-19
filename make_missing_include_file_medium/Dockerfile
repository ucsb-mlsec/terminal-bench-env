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
    apt-get install -y make && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/pyproject/src /home/pyproject/tests /home/pyproject/config /home/pyproject/build /tmp

COPY src/calculator.py /home/pyproject/src/calculator.py
COPY tests/test_calculator.py /home/pyproject/tests/test_calculator.py
COPY Makefile /home/pyproject/Makefile
COPY config/python.mk /home/pyproject/config/python.mk
COPY config/testing.mk /home/pyproject/config/testing.mk
COPY config/packaging.mk /home/pyproject/config/packaging.mk

WORKDIR /home/pyproject