FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/cp2k_inputs

COPY data/cp2k_inputs/valid_water.inp /workspace/cp2k_inputs/valid_water.inp
COPY data/cp2k_inputs/missing_basis_set.inp /workspace/cp2k_inputs/missing_basis_set.inp
COPY data/cp2k_inputs/empty_basis_value.inp /workspace/cp2k_inputs/empty_basis_value.inp
COPY data/cp2k_inputs/mismatched_tags.inp /workspace/cp2k_inputs/mismatched_tags.inp
COPY data/cp2k_inputs/duplicate_kinds.inp /workspace/cp2k_inputs/duplicate_kinds.inp
COPY data/cp2k_inputs/valid_methane.inp /workspace/cp2k_inputs/valid_methane.inp
COPY data/cp2k_inputs/multiple_issues.inp /workspace/cp2k_inputs/multiple_issues.inp
COPY data/cp2k_inputs/valid_complex.inp /workspace/cp2k_inputs/valid_complex.inp
COPY data/cp2k_inputs/unclosed_kind.inp /workspace/cp2k_inputs/unclosed_kind.inp
COPY data/cp2k_inputs/empty_basis_multiple.inp /workspace/cp2k_inputs/empty_basis_multiple.inp
COPY data/cp2k_inputs/valid_silicon.inp /workspace/cp2k_inputs/valid_silicon.inp
COPY data/cp2k_inputs/triple_duplicate.inp /workspace/cp2k_inputs/triple_duplicate.inp
COPY data/cp2k_inputs/missing_and_empty.inp /workspace/cp2k_inputs/missing_and_empty.inp
COPY data/cp2k_inputs/valid_ammonia.inp /workspace/cp2k_inputs/valid_ammonia.inp
COPY data/cp2k_inputs/wrong_end_tag.inp /workspace/cp2k_inputs/wrong_end_tag.inp

WORKDIR /workspace