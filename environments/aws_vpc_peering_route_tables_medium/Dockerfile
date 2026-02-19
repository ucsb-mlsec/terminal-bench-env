FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /opt/aws-config /tmp

COPY config/vpc_a_config.json /opt/aws-config/vpc_a_config.json
COPY config/vpc_b_config.json /opt/aws-config/vpc_b_config.json
COPY config/route_table_vpc_a.json /opt/aws-config/route_table_vpc_a.json
COPY config/route_table_vpc_b.json /opt/aws-config/route_table_vpc_b.json
COPY config/peering_connection.json /opt/aws-config/peering_connection.json

WORKDIR /workspace