FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /home/user/iam_policies

COPY data/iam_policies/dev-user-policy.json /home/user/iam_policies/dev-user-policy.json
COPY data/iam_policies/s3-readonly-policy.json /home/user/iam_policies/s3-readonly-policy.json
COPY data/iam_policies/admin-role-trust-policy.json /home/user/iam_policies/admin-role-trust-policy.json
COPY data/iam_policies/admin-role-permissions.json /home/user/iam_policies/admin-role-permissions.json
COPY data/iam_policies/ec2-basic-policy.json /home/user/iam_policies/ec2-basic-policy.json
COPY data/iam_policies/cloudwatch-logs-policy.json /home/user/iam_policies/cloudwatch-logs-policy.json
COPY data/iam_policies/user_policies.txt /home/user/iam_policies/user_policies.txt

WORKDIR /home/user