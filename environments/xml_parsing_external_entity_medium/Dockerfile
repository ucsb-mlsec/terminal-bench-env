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

RUN pip3 install --no-cache-dir --break-system-packages lxml pytest

RUN mkdir -p /app/parsers /app/tests /app/sample_data /solution

COPY app/parsers/etree_parser.py /app/parsers/etree_parser.py
COPY app/parsers/lxml_parser.py /app/parsers/lxml_parser.py
COPY app/parsers/minidom_parser.py /app/parsers/minidom_parser.py
COPY app/tests/__init__.py /app/tests/__init__.py
COPY app/tests/test_security.py /app/tests/test_security.py
COPY app/sample_data/valid.xml /app/sample_data/valid.xml
COPY app/sample_data/malicious_xxe.xml /app/sample_data/malicious_xxe.xml
COPY app/sample_data/entity_expansion.xml /app/sample_data/entity_expansion.xml

WORKDIR /app