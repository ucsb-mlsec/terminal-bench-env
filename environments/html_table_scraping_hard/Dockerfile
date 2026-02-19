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
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages beautifulsoup4 lxml

RUN mkdir -p /workspace/financial_reports /workspace/output

COPY data/financial_reports/Q1_2020.html /workspace/financial_reports/Q1_2020.html
COPY data/financial_reports/Q2_2020.html /workspace/financial_reports/Q2_2020.html
COPY data/financial_reports/Q3_2020.html /workspace/financial_reports/Q3_2020.html
COPY data/financial_reports/Q4_2020.html /workspace/financial_reports/Q4_2020.html
COPY data/financial_reports/Q1_2021.html /workspace/financial_reports/Q1_2021.html
COPY data/financial_reports/Q2_2021.html /workspace/financial_reports/Q2_2021.html
COPY data/financial_reports/Q3_2021.html /workspace/financial_reports/Q3_2021.html
COPY data/financial_reports/Q4_2021.html /workspace/financial_reports/Q4_2021.html
COPY data/financial_reports/Q1_2022.html /workspace/financial_reports/Q1_2022.html
COPY data/financial_reports/Q2_2022.html /workspace/financial_reports/Q2_2022.html

WORKDIR /workspace