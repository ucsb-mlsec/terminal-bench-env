FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

# Create necessary directory structures
RUN mkdir -p /opt/backup_data/dir1 \
    /opt/backup_data/dir2 \
    /opt/backup_data/dir3 \
    /opt/backup_data/dir4 \
    /opt/backup_data/dir5 \
    /opt/backup_data/dir6 \
    /opt/backup_data/projects/project_a \
    /opt/backup_data/projects/shared \
    /opt/backup_data/archive \
    /opt/backup_data/self_loop \
    /tmp

# Copy regular files
COPY opt/backup_data/dir1/file1.txt /opt/backup_data/dir1/file1.txt
COPY opt/backup_data/dir2/file2.txt /opt/backup_data/dir2/file2.txt
COPY opt/backup_data/dir3/file3.txt /opt/backup_data/dir3/file3.txt
COPY opt/backup_data/dir4/data.log /opt/backup_data/dir4/data.log
COPY opt/backup_data/dir5/backup.dat /opt/backup_data/dir5/backup.dat
COPY opt/backup_data/dir6/readme.txt /opt/backup_data/dir6/readme.txt
COPY opt/backup_data/projects/project_a/code.py /opt/backup_data/projects/project_a/code.py
COPY opt/backup_data/projects/shared/resources.txt /opt/backup_data/projects/shared/resources.txt
COPY opt/backup_data/archive/old_backups.tar.gz /opt/backup_data/archive/old_backups.tar.gz
COPY opt/backup_data/self_loop/normal_file.txt /opt/backup_data/self_loop/normal_file.txt

# Create symbolic links for circular chains
# Loop 1: dir1/link_a -> dir2, dir2/link_b -> dir1
RUN ln -s /opt/backup_data/dir2 /opt/backup_data/dir1/link_a && \
    ln -s /opt/backup_data/dir1 /opt/backup_data/dir2/link_b

# Loop 2: dir3 -> dir4 -> dir5 -> dir3
RUN ln -s /opt/backup_data/dir4 /opt/backup_data/dir3/link_to_dir4 && \
    ln -s /opt/backup_data/dir5 /opt/backup_data/dir4/link_to_dir5 && \
    ln -s /opt/backup_data/dir3 /opt/backup_data/dir5/link_to_dir3

# Loop 3: projects/project_a -> shared, shared -> project_a
RUN ln -s /opt/backup_data/projects/shared /opt/backup_data/projects/project_a/link_to_shared && \
    ln -s /opt/backup_data/projects/project_a /opt/backup_data/projects/shared/link_back

# Loop 4: self_loop (self-referencing)
RUN ln -s /opt/backup_data/self_loop /opt/backup_data/self_loop/link_to_self

WORKDIR /opt/backup_data