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
    apt-get install -y jq && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/user /workspace/config /workspace/data

COPY config/lambda_role_policy.json /workspace/config/lambda_role_policy.json
COPY config/lambda_role_trust.json /workspace/config/lambda_role_trust.json
COPY config/ec2_role_policy.json /workspace/config/ec2_role_policy.json
COPY config/ec2_role_trust.json /workspace/config/ec2_role_trust.json
COPY config/replication_role_policy.json /workspace/config/replication_role_policy.json
COPY config/replication_role_trust.json /workspace/config/replication_role_trust.json
COPY config/analytics_role_policy.json /workspace/config/analytics_role_policy.json
COPY config/analytics_role_trust.json /workspace/config/analytics_role_trust.json
COPY config/kms_key_policy.json /workspace/config/kms_key_policy.json
COPY data/s3_access_logs.txt /workspace/data/s3_access_logs.txt

WORKDIR /workspace