#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// Define parameters
params.input_dir = '/workspace/input_data'
params.output_dir = '/workspace/results'
params.quality_threshold = 20

// Create channel from input FASTQ files
fastq_files = Channel.fromPath("${params.input_dir}/*.fastq")
                     .ifEmpty { error "No FASTQ files found in ${params.input_dir}" }

// Process to perform quality check on FASTQ files
process qualityCheck {
    publishDir params.output_dir, mode: 'copy'
    
    input:
    path fastq_file
    
    output:
    path "${fastq_file.baseName}_qc_results.txt"
    
    script:
    """
    awk -v threshold=${params.quality_threshold} '
    BEGIN { 
        total_seqs = 0
        passing_seqs = 0
    }
    NR % 4 == 2 { 
        seq_length = length(\$0)
    }
    NR % 4 == 0 {
        total_seqs++
        quality_line = \$0
        sum = 0
        for (i = 1; i <= length(quality_line); i++) {
            char = substr(quality_line, i, 1)
            phred = (ord(char) - 33)
            sum += phred
        }
        avg_quality = sum / length(quality_line)
        if (avg_quality > threshold) {
            passing_seqs++
        }
    }
    END {
        print "Total sequences: " total_seqs
        print "Sequences passing QC (avg quality >" threshold "): " passing_seqs
        print "Pass rate: " (passing_seqs/total_seqs)*100 "%"
    }
    function ord(char) {
        return index(" !\"#\$%&'\''()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~", char) + 31
    }
    ' ${fastq_file} > ${fastq_file.baseName}_qc_results.txt
    """
}

// Workflow definition
workflow {
    qualityCheck(fastq_files
}