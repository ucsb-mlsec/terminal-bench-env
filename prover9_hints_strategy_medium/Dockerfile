FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/hints

COPY hints/valid_basic.p9 /workspace/hints/valid_basic.p9
COPY hints/valid_complex.p9 /workspace/hints/valid_complex.p9
COPY hints/missing_header.p9 /workspace/hints/missing_header.p9
COPY hints/missing_footer.p9 /workspace/hints/missing_footer.p9
COPY hints/missing_periods.p9 /workspace/hints/missing_periods.p9
COPY hints/unbalanced_parens.p9 /workspace/hints/unbalanced_parens.p9
COPY hints/empty_hints.p9 /workspace/hints/empty_hints.p9
COPY hints/invalid_label.p9 /workspace/hints/invalid_label.p9
COPY hints/valid_with_comments.p9 /workspace/hints/valid_with_comments.p9
COPY hints/wrong_case_header.p9 /workspace/hints/wrong_case_header.p9
COPY hints/wrong_case_footer.p9 /workspace/hints/wrong_case_footer.p9
COPY hints/valid_all_operators.p9 /workspace/hints/valid_all_operators.p9
COPY hints/multiple_errors.p9 /workspace/hints/multiple_errors.p9

WORKDIR /workspace