FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /data/payroll_archive /data/format_specs /output

COPY data/payroll_archive/payroll_2021_q1.txt /data/payroll_archive/payroll_2021_q1.txt
COPY data/payroll_archive/payroll_2021_q2.txt /data/payroll_archive/payroll_2021_q2.txt
COPY data/payroll_archive/payroll_2022_h1.txt /data/payroll_archive/payroll_2022_h1.txt
COPY data/payroll_archive/payroll_2022_h2.txt /data/payroll_archive/payroll_2022_h2.txt
COPY data/payroll_archive/payroll_2023_q1.txt /data/payroll_archive/payroll_2023_q1.txt
COPY data/format_specs/format_v1_partial.txt /data/format_specs/format_v1_partial.txt
COPY data/format_specs/format_v2_partial.txt /data/format_specs/format_v2_partial.txt
COPY data/format_specs/format_v3_notes.txt /data/format_specs/format_v3_notes.txt
COPY data/format_specs/format_v4_fragment.txt /data/format_specs/format_v4_fragment.txt

WORKDIR /workspace