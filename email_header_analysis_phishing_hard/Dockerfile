FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/emails

COPY data/emails/email_001.eml /workspace/emails/email_001.eml
COPY data/emails/email_002.eml /workspace/emails/email_002.eml
COPY data/emails/email_003.eml /workspace/emails/email_003.eml
COPY data/emails/email_004.eml /workspace/emails/email_004.eml
COPY data/emails/email_005.eml /workspace/emails/email_005.eml
COPY data/emails/email_006.eml /workspace/emails/email_006.eml
COPY data/emails/email_007.eml /workspace/emails/email_007.eml
COPY data/emails/email_008.eml /workspace/emails/email_008.eml
COPY data/emails/email_009.eml /workspace/emails/email_009.eml
COPY data/emails/email_010.eml /workspace/emails/email_010.eml
COPY data/emails/email_011.eml /workspace/emails/email_011.eml
COPY data/emails/email_012.eml /workspace/emails/email_012.eml
COPY data/emails/email_013.eml /workspace/emails/email_013.eml
COPY data/emails/email_014.eml /workspace/emails/email_014.eml
COPY data/emails/email_015.eml /workspace/emails/email_015.eml

WORKDIR /workspace