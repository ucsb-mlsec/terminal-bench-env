version 1.0

workflow FastaStats {
  input {
    Array[File] fasta_files
  

  call CountSequences {
    input:
      files = fasta_file
  }

  call CalculateBases {
    input:
      files = fasta_files,
      counts = CountSequences.count
  }

  output {
    File stats = CalculateBases.statistics
  }
}

task CountSequences {
  input {
    Array[File] files
  }

  command {
    count=0
    for file in ${sep=' ' files}; do
      sequences=$(grep -c "^>" $file)
      count=$((count + sequences))
    done
    echo $count > sequence_count.txt
  }

  output {
    Int count = read_int("sequence_count.txt")
  }

  runtime {
    docker: "ubuntu:latest"
  }
}

task CalculateBases {
  input {
    Array[File] files
    Int counts
  }

  command <<<
    total_bases=0
    for file in ~{sep=' ' files}; do
      bases=$(grep -v "^>" $file | tr -d '\n' | wc -c
      total_bases=$((total_bases + bases))
    done
    
    echo "sequences=~{counts}" > stats.txt
    echo "total_bases=$total_bases" >> stats.txt
    
    cp stats.txt /workspace/results/stats.txt
  >>>

  output {
    File statistics = "stats.txt"
  }

  runtime {
    docker: "ubuntu:latest"

}