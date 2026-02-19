FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/iam_roles

COPY iam_roles/production-api-role.json /workspace/iam_roles/production-api-role.json
COPY iam_roles/developer-access-role.json /workspace/iam_roles/developer-access-role.json
COPY iam_roles/cross-account-admin.json /workspace/iam_roles/cross-account-admin.json
COPY iam_roles/audit-role.json /workspace/iam_roles/audit-role.json
COPY iam_roles/backup-service-role.json /workspace/iam_roles/backup-service-role.json
COPY iam_roles/dev-environment-role.json /workspace/iam_roles/dev-environment-role.json
COPY iam_roles/partner-integration-role.json /workspace/iam_roles/partner-integration-role.json
COPY iam_roles/s3-replication-role.json /workspace/iam_roles/s3-replication-role.json

WORKDIR /workspace