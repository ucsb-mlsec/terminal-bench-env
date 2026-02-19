FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from Phase 2.5 analysis
RUN apt-get update && apt-get install -y rsync && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/legacy_app/data/module_a \
    /opt/legacy_app/data/module_b \
    /opt/legacy_app/data/module_c \
    /opt/legacy_app/data/module_d \
    /opt/legacy_app/data/shared \
    /opt/legacy_app/data/common \
    /opt/legacy_app/data/archive \
    /backup \
    /solution

COPY data/legacy_app/data/module_a/config.ini /opt/legacy_app/data/module_a/config.ini
COPY data/legacy_app/data/module_a/shared_lib.so /opt/legacy_app/data/module_a/shared_lib.so
COPY data/legacy_app/data/module_a/data_v1.dat /opt/legacy_app/data/module_a/data_v1.dat
COPY data/legacy_app/data/module_b/config.ini /opt/legacy_app/data/module_b/config.ini
COPY data/legacy_app/data/module_b/shared_lib.so /opt/legacy_app/data/module_b/shared_lib.so
COPY data/legacy_app/data/module_b/module_b_specific.txt /opt/legacy_app/data/module_b/module_b_specific.txt
COPY data/legacy_app/data/module_c/config.ini /opt/legacy_app/data/module_c/config.ini
COPY data/legacy_app/data/module_c/shared_lib.so /opt/legacy_app/data/module_c/shared_lib.so
COPY data/legacy_app/data/module_c/data_v1.dat /opt/legacy_app/data/module_c/data_v1.dat
COPY data/legacy_app/data/shared/common_config.xml /opt/legacy_app/data/shared/common_config.xml
COPY data/legacy_app/data/shared/reference_data.json /opt/legacy_app/data/shared/reference_data.json
COPY data/legacy_app/data/common/common_config.xml /opt/legacy_app/data/common/common_config.xml
COPY data/legacy_app/data/common/reference_data.json /opt/legacy_app/data/common/reference_data.json
COPY data/legacy_app/data/common/common_lib.so /opt/legacy_app/data/common/common_lib.so
COPY data/legacy_app/data/archive/data_v1.dat /opt/legacy_app/data/archive/data_v1.dat
COPY data/legacy_app/data/archive/old_config.ini /opt/legacy_app/data/archive/old_config.ini
COPY data/legacy_app/data/module_d/unique_file.log /opt/legacy_app/data/module_d/unique_file.log
COPY data/legacy_app/data/module_d/standalone.dat /opt/legacy_app/data/module_d/standalone.dat

RUN ln -f /opt/legacy_app/data/module_a/config.ini /opt/legacy_app/data/module_b/config.ini && \
    ln -f /opt/legacy_app/data/module_a/config.ini /opt/legacy_app/data/module_c/config.ini && \
    ln -f /opt/legacy_app/data/module_a/config.ini /opt/legacy_app/data/archive/old_config.ini && \
    ln -f /opt/legacy_app/data/module_a/shared_lib.so /opt/legacy_app/data/module_b/shared_lib.so && \
    ln -f /opt/legacy_app/data/module_a/shared_lib.so /opt/legacy_app/data/module_c/shared_lib.so && \
    ln -f /opt/legacy_app/data/module_a/data_v1.dat /opt/legacy_app/data/module_c/data_v1.dat && \
    ln -f /opt/legacy_app/data/module_a/data_v1.dat /opt/legacy_app/data/archive/data_v1.dat && \
    ln -f /opt/legacy_app/data/shared/common_config.xml /opt/legacy_app/data/common/common_config.xml && \
    ln -f /opt/legacy_app/data/shared/reference_data.json /opt/legacy_app/data/common/reference_data.json

WORKDIR /workspace