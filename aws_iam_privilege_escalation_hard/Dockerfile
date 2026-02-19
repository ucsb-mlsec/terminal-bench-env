FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /audit/aws_config /solution

COPY data/users.json /audit/aws_config/users.json
COPY data/roles.json /audit/aws_config/roles.json
COPY data/policies.json /audit/aws_config/policies.json
COPY data/policy_documents.json /audit/aws_config/policy_documents.json
COPY data/inline_policies.json /audit/aws_config/inline_policies.json
COPY data/role_trust_relationships.json /audit/aws_config/role_trust_relationships.json

WORKDIR /workspace