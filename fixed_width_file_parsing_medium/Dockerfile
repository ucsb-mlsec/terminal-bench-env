FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/payroll/specs /output

COPY data/payroll/hr_records.txt /data/payroll/hr_records.txt
COPY data/payroll/finance_records.txt /data/payroll/finance_records.txt
COPY data/payroll/timekeeping_records.txt /data/payroll/timekeeping_records.txt
COPY data/payroll/specs/hr_records.spec /data/payroll/specs/hr_records.spec
COPY data/payroll/specs/finance_records.spec /data/payroll/specs/finance_records.spec
COPY data/payroll/specs/timekeeping_records.spec /data/payroll/specs/timekeeping_records.spec

WORKDIR /workspace