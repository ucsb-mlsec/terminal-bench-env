FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/crawler/data

COPY url_tracker.py /opt/crawler/url_tracker.py
COPY crawl_simulator.py /opt/crawler/crawl_simulator.py
COPY data/urls_dataset.txt /opt/crawler/urls_dataset.txt
COPY data/test_queries.txt /opt/crawler/test_queries.txt

WORKDIR /opt/crawler