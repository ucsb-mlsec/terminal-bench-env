FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data /home/user

COPY data/V1__initial_schema.sql /data/V1__initial_schema.sql
COPY data/V2__add_user_email.sql /data/V2__add_user_email.sql
COPY data/V2__add_product_category.sql /data/V2__add_product_category.sql
COPY data/V3__create_orders_table.sql /data/V3__create_orders_table.sql
COPY data/V5__add_order_status.sql /data/V5__add_order_status.sql
COPY data/V5__add_user_phone.sql /data/V5__add_user_phone.sql
COPY data/V6__create_payment_table.sql /data/V6__create_payment_table.sql
COPY data/V3__add_product_price.sql /data/V3__add_product_price.sql
COPY data/V7__add_product_description.sql /data/V7__add_product_description.sql
COPY data/V8__add_user_created_at.sql /data/V8__add_user_created_at.sql
COPY data/flyway_schema_history.csv /data/flyway_schema_history.csv

WORKDIR /home/user