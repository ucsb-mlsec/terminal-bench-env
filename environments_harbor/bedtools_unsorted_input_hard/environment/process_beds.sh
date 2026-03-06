#!/bin/bash
set -e

# Define BED file paths
PEAKS1="/workspace/bed_files/peaks_sample1.bed"
PEAKS2="/workspace/bed_files/peaks_sample2.bed"
REGIONS_CONTROL="/workspace/bed_files/regions_control.bed"
REGIONS_TEST="/workspace/bed_files/regions_test.bed"

echo "Starting BED file processing..."

# Find overlaps between peaks_sample1.bed and regions_control.bed
echo "Finding overlaps between sample1 peaks and control regions..."
bedtools intersect -a "$PEAKS1" -b "$REGIONS_CONTROL" > /tmp/overlap1.bed
echo "Overlap1 completed successfully!"

# Find overlaps between peaks_sample2.bed and regions_test.bed
echo "Finding overlaps between sample2 peaks and test regions..."
bedtools intersect -a "$PEAKS2" -b "$REGIONS_TEST" > /tmp/overlap2.bed
echo "Overlap2 completed successfully!"

# Merge overlapping intervals in peaks_sample1.bed
echo "Merging overlapping intervals in sample1 peaks..."
bedtools merge -i "$PEAKS1" > /tmp/merged1.bed
echo "Merge1 completed successfully!"

# Merge overlapping intervals in peaks_sample2.bed
echo "Merging overlapping intervals in sample2 peaks..."
bedtools merge -i "$PEAKS2" > /tmp/merged2.bed
echo "Merge2 completed successfully!"

echo "All operations completed successfully!"