FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/migrations /tmp

COPY workspace/flyway_schema_history.txt /workspace/flyway_schema_history.txt
COPY workspace/migrations/V1.0.0__create_users_table.sql /workspace/migrations/V1.0.0__create_users_table.sql
COPY workspace/migrations/V1.1.0__add_email_column.sql /workspace/migrations/V1.1.0__add_email_column.sql
COPY workspace/migrations/V2.0.0__create_orders_table.sql /workspace/migrations/V2.0.0__create_orders_table.sql
COPY workspace/migrations/V2.3.0__add_index_customer.sql /workspace/migrations/V2.3.0__add_index_customer.sql
COPY workspace/migrations/V2.3.3__add_product_table.sql /workspace/migrations/V2.3.3__add_product_table.sql
COPY workspace/migrations/V2.3.5__add_category_column.sql /workspace/migrations/V2.3.5__add_category_column.sql
COPY workspace/migrations/V2.3.5__modify_orders_status.sql /workspace/migrations/V2.3.5__modify_orders_status.sql
COPY workspace/migrations/V2.4.0__add_shipping_table.sql /workspace/migrations/V2.4.0__add_shipping_table.sql
COPY workspace/migrations/V2.3.6__add_payment_table.sql /workspace/migrations/V2.3.6__add_payment_table.sql
COPY workspace/migrations/V2.3.0__create_audit_log.sql /workspace/migrations/V2.3.0__create_audit_log.sql

WORKDIR /workspace