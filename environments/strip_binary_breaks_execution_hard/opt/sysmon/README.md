# System Monitor (sysmon)

A lightweight system monitoring agent designed to collect and display essential system metrics. This utility provides real-time information about system resource usage, performance statistics, and overall health status. It is designed to run with minimal overhead and can be easily integrated into automated monitoring workflows.

## Requirements

- GCC compiler (gcc)
- Standard C library development files
- Make build system

## Building

To compile the system monitor from source:

```bash
make
```

This will compile the source code and produce the `sysmon` binary in the current directory.

## Installation

To install the compiled binary to the system directory:

```bash
sudo make install
```

This will copy the binary to `/usr/local/bin/sysmon` and set the appropriate permissions.

## Usage

Once installed, you can run the system monitor using:

```bash
/usr/local/bin/sysmon
```

Or simply:

```bash
sysmon
```

if `/usr/local/bin` is in your PATH.

## Output

The system monitor displays the following information:
- System uptime and current timestamp
- CPU usage statistics
- Memory utilization (used, free, total)
- Disk I/O metrics
- Network interface status
- Overall system health status

The output is formatted for easy readability and can be redirected or parsed by other monitoring tools.