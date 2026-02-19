FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /app /app/test_data/legitimate /app/test_data/attacks /solution

RUN pip3 install --no-cache-dir --break-system-packages defusedxml lxml

COPY app/data_importer.py /app/data_importer.py
COPY app/config_loader.py /app/config_loader.py
COPY app/api_handler.py /app/api_handler.py
COPY app/test_data/legitimate/sample1.xml /app/test_data/legitimate/sample1.xml
COPY app/test_data/legitimate/config.xml /app/test_data/legitimate/config.xml
COPY app/test_data/legitimate/api_request.xml /app/test_data/legitimate/api_request.xml
COPY app/test_data/attacks/xxe_file_read.xml /app/test_data/attacks/xxe_file_read.xml
COPY app/test_data/attacks/xxe_ssrf.xml /app/test_data/attacks/xxe_ssrf.xml
COPY app/test_data/attacks/xxe_parameter_entity.xml /app/test_data/attacks/xxe_parameter_entity.xml
COPY app/test_data/attacks/billion_laughs.xml /app/test_data/attacks/billion_laughs.xml

WORKDIR /app