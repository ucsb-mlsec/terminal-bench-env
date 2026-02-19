#!/bin/bash

# Simple genomic intersection analysis
# Find overlaps between genes and peaks

bedtools intersect -a genes.bed -b peaks.bed > overlaps.bed