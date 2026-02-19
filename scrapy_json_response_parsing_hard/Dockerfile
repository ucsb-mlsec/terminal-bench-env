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

RUN pip3 install --no-cache-dir --break-system-packages scrapy

RUN mkdir -p /workspace/feeds /workspace/output /workspace/feed_processor/feed_processor/spiders

COPY data/partner_a.json /workspace/feeds/partner_a.json
COPY data/partner_b.json /workspace/feeds/partner_b.json
COPY data/partner_c.json /workspace/feeds/partner_c.json
COPY feed_processor/scrapy.cfg /workspace/feed_processor/scrapy.cfg
COPY feed_processor/feed_processor/items.py /workspace/feed_processor/feed_processor/items.py
COPY feed_processor/feed_processor/spiders/__init__.py /workspace/feed_processor/feed_processor/spiders/__init__.py
COPY feed_processor/feed_processor/spiders/feed_spider.py /workspace/feed_processor/feed_processor/spiders/feed_spider.py
COPY feed_processor/feed_processor/pipelines.py /workspace/feed_processor/feed_processor/pipelines.py
COPY feed_processor/feed_processor/middlewares.py /workspace/feed_processor/feed_processor/middlewares.py

WORKDIR /workspace