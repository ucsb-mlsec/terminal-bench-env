FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /emails /solution

COPY data/email1.eml /emails/email1.eml
COPY data/email2.eml /emails/email2.eml
COPY data/email3.eml /emails/email3.eml
COPY data/email4.eml /emails/email4.eml
COPY data/email5.eml /emails/email5.eml
COPY data/email6.eml /emails/email6.eml
COPY data/email7.eml /emails/email7.eml
COPY data/email8.eml /emails/email8.eml

WORKDIR /workspace