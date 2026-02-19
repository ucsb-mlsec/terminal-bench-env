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

RUN pip3 install --no-cache-dir --break-system-packages bleach html5lib lxml pyyaml

RUN mkdir -p /opt/cms/data/samples/legitimate /opt/cms/data/samples/malicious /opt/cms/config

COPY sanitizer.py /opt/cms/sanitizer.py
COPY test_xss.py /opt/cms/test_xss.py
COPY data/samples/legitimate/article1.html /opt/cms/data/samples/legitimate/article1.html
COPY data/samples/legitimate/blog_post.html /opt/cms/data/samples/legitimate/blog_post.html
COPY data/samples/legitimate/tutorial.html /opt/cms/data/samples/legitimate/tutorial.html
COPY data/samples/malicious/basic_xss.html /opt/cms/data/samples/malicious/basic_xss.html
COPY data/samples/malicious/obfuscated_xss.html /opt/cms/data/samples/malicious/obfuscated_xss.html
COPY config/cms_config.yaml /opt/cms/config/cms_config.yaml
COPY content_processor.py /opt/cms/content_processor.py

WORKDIR /opt/cms