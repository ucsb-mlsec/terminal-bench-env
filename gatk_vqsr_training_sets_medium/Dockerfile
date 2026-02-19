FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive

# Install python3-pip and create python symlink
RUN apt-get update && \
    apt-get install -y python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install test dependencies
RUN pip3 install --no-cache-dir --break-system-packages pytest==8.4.1 pandas==2.3.2 scipy==1.16.1

RUN mkdir -p /workspace/variant_callsets /workspace/solution

COPY data/dbsnp_filtered.vcf /workspace/variant_callsets/dbsnp_filtered.vcf
COPY data/dbsnp_filtered.vcf.stats /workspace/variant_callsets/dbsnp_filtered.vcf.stats
COPY data/hapmap_variants.vcf /workspace/variant_callsets/hapmap_variants.vcf
COPY data/hapmap_variants.vcf.stats /workspace/variant_callsets/hapmap_variants.vcf.stats
COPY data/low_quality_set.vcf /workspace/variant_callsets/low_quality_set.vcf
COPY data/low_quality_set.vcf.stats /workspace/variant_callsets/low_quality_set.vcf.stats
COPY data/small_callset.vcf /workspace/variant_callsets/small_callset.vcf
COPY data/small_callset.vcf.stats /workspace/variant_callsets/small_callset.vcf.stats
COPY data/poor_titv_ratio.vcf /workspace/variant_callsets/poor_titv_ratio.vcf
COPY data/poor_titv_ratio.vcf.stats /workspace/variant_callsets/poor_titv_ratio.vcf.stats
COPY data/imbalanced_genotypes.vcf /workspace/variant_callsets/imbalanced_genotypes.vcf
COPY data/imbalanced_genotypes.vcf.stats /workspace/variant_callsets/imbalanced_genotypes.vcf.stats
COPY data/platinum_genomes.vcf /workspace/variant_callsets/platinum_genomes.vcf
COPY data/platinum_genomes.vcf.stats /workspace/variant_callsets/platinum_genomes.vcf.stats
COPY data/exome_variants.vcf /workspace/variant_callsets/exome_variants.vcf
COPY data/exome_variants.vcf.stats /workspace/variant_callsets/exome_variants.vcf.stats

WORKDIR /workspace