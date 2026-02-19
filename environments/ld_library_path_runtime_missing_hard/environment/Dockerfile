FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/dataprocessor \
    && mkdir -p /usr/local/lib/custom_modules \
    && mkdir -p /var/lib/python_modules \
    && mkdir -p /home/datauser/python_libs \
    && mkdir -p /opt/shared/python \
    && mkdir -p /solution

COPY opt/dataprocessor/main.py /opt/dataprocessor/main.py
COPY usr/local/lib/custom_modules/data_loader.py /usr/local/lib/custom_modules/data_loader.py
COPY var/lib/python_modules/data_transformer.py /var/lib/python_modules/data_transformer.py
COPY home/datauser/python_libs/data_validator.py /home/datauser/python_libs/data_validator.py
COPY opt/shared/python/report_generator.py /opt/shared/python/report_generator.py

WORKDIR /solution

# Mapped from docker-compose.yaml
ENV TEST_DIR=/tests
