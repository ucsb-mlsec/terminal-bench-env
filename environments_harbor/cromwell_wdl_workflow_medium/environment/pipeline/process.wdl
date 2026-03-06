version 1.0

workflow ProcessLogs {
  input {
    Array[File] log_files
    String filter_pattern = "ERROR"
    String output_prefix = "processed"
  }

  scatter (log_file in log_files) {
    call FilterLogs {
      input:
        input_log = log_file,
        pattern = filter_pattern
    }
    
    call TransformLogs {
      input:
        filtered_log = FilterLogs.filtered_output,
        prefix = output_prefix
    }
  }

  call MergeLogs {
    input:
      transformed_logs = TransformLogs.transformed_output,
      output_name = output_prefix + "_merged.txt"
  }

  output {
    Array[File] filtered_logs = FilterLogs.filtered_output
    Array[File] transformed_logs = TransformLogs.transformed_output
    File merged_log = MergeLogs.merged_output
  }
}

task FilterLogs {
  input {
    File input_log
    String pattern
  }

  command <<<
    grep "~{pattern}" ~{input_log} > filtered_~{basename(input_log)} || touch filtered_~{basename(input_log)}
  >>>

  output {
    File filtered_output = "filtered_" + basename(input_log)
  }

  runtime {
    docker: "ubuntu:20.04"
    memory: "2 GB"
    cpu: 1
  }
}

task TransformLogs {
  input {
    File filtered_log
    String prefix
  }

  command <<<
    awk '{print toupper($0)}' ~{filtered_log} > ~{prefix}_~{basename(filtered_log)}
  >>>

  output {
    File transformed_output = prefix + "_" + basename(filtered_log)
  }

  runtime {
    docker: "ubuntu:20.04"
    memory: "2 GB"
    cpu: 1
  }
}

task MergeLogs {
  input {
    Array[File] transformed_logs
    String output_name
  }

  command <<<
    cat ~{sep=' ' transformed_logs} > ~{output_name}
    echo "Total lines processed: $(wc -l < ~{output_name})" >> ~{output_name}
  >>>

  output {
    File merged_output = output_name
  }

  runtime {
    docker: "ubuntu:20.04"
    memory: "2 GB"
    cpu: 1
  }
}