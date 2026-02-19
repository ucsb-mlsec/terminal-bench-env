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
    apt-get install -y python3-pip libxml2-dev libxslt1-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages lxml signxml cryptography

RUN mkdir -p /opt/saml

COPY opt/saml/verifier.py /opt/saml/verifier.py
COPY opt/saml/idp_cert.pem /opt/saml/idp_cert.pem
COPY opt/saml/sample_assertion.xml /opt/saml/sample_assertion.xml

RUN chmod +x /opt/saml/verifier.py

WORKDIR /opt/saml