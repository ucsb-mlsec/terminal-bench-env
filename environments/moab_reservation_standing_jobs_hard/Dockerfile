FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /cluster/requests \
    && mkdir -p /etc/moab \
    && mkdir -p /cluster \
    && mkdir -p /etc \
    && mkdir -p /root

COPY data/bio_lab_request.txt /cluster/requests/bio_lab_request.txt
COPY config/moab.cfg /etc/moab/moab.cfg
COPY config/standing-reservations.cfg /etc/moab/standing-reservations.cfg
COPY config/access-control.cfg /etc/moab/access-control.cfg
COPY data/active_reservations.log /cluster/active_reservations.log
COPY data/group /etc/group

WORKDIR /root