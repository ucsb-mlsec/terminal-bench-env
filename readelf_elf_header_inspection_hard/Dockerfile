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
    apt-get install -y \
    binutils \
    file \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages pyelftools

RUN mkdir -p /opt/production/bin/subdir /tmp

COPY data/valid_binary_x86_64 /opt/production/bin/valid_binary_x86_64
COPY data/valid_binary_arm /opt/production/bin/valid_binary_arm
COPY data/valid_binary_mips /opt/production/bin/valid_binary_mips
COPY data/valid_binary_aarch64 /opt/production/bin/valid_binary_aarch64
COPY data/corrupted_null_entrypoint /opt/production/bin/corrupted_null_entrypoint
COPY data/corrupted_invalid_machine /opt/production/bin/corrupted_invalid_machine
COPY data/corrupted_bad_class /opt/production/bin/corrupted_bad_class
COPY data/suspicious_low_entrypoint /opt/production/bin/suspicious_low_entrypoint
COPY data/not_elf_text_file.txt /opt/production/bin/not_elf_text_file.txt
COPY data/not_elf_script.sh /opt/production/bin/not_elf_script.sh
COPY data/corrupted_magic_partial /opt/production/bin/corrupted_magic_partial
COPY data/subdir/valid_nested_x86 /opt/production/bin/subdir/valid_nested_x86
COPY data/subdir/corrupted_nested /opt/production/bin/subdir/corrupted_nested
COPY data/valid_powerpc /opt/production/bin/valid_powerpc
COPY data/suspicious_mismatch /opt/production/bin/suspicious_mismatch

WORKDIR /opt/production/bin