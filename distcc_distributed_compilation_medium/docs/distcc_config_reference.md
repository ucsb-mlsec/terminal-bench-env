# distcc Configuration Reference

## 1. Client Configuration (DISTCC_HOSTS format)

The distcc client configuration specifies which compilation hosts to use and how many concurrent jobs each should handle. This configuration can be set in two ways:

- **Environment Variable**: Set `DISTCC_HOSTS` environment variable
- **Configuration File**: Create `~/.distcc/hosts` or `/etc/distcc/hosts`

### Format Specification

The basic format for specifying hosts is:

```
hostname[/limit][,options][@host_options]
```

Where:
- `hostname`: IP address or hostname of the compilation server
- `/limit`: Maximum number of concurrent compilation jobs to send to this host (optional)
- `,options`: Additional options like `cpp` (preprocess remotely) or `lzo` (compression)
- `@host_options`: Host-specific options

### Limit Specification

The `/limit` parameter is crucial for controlling job distribution:
- `192.168.1.10/6` - Use host 192.168.1.10 with up to 6 concurrent jobs
- `localhost/4` - Use local machine with 4 concurrent jobs
- `node1.build.local/8` - Use hostname with 8 concurrent jobs

### Multiple Hosts

Separate multiple hosts with spaces:

```
localhost/4 192.168.1.10/8 192.168.1.11/6 192.168.1.12/4
```

### Default Port

The default distcc port is **3632**. If a server runs on a different port, specify it with a colon:

```
192.168.1.10:3633/8
```

### Complete Examples

**Example 1: Simple configuration**
```
localhost/4 192.168.1.10/8 192.168.1.11/6
```

**Example 2: With compression**
```
localhost/4 192.168.1.10,lzo/8 192.168.1.11,lzo/6
```

**Example 3: Custom port**
```
localhost/4 192.168.1.10:4000/8 192.168.1.11/6
```

## 2. Server Configuration (Access Control)

distcc servers must be configured to control which clients can connect and submit compilation jobs. This is critical for security.

### Access Control Methods

Access control is configured using:
- **DISTCC_ALLOW_NETS environment variable**: Set allowed networks
- **--allow command-line option**: Specify allowed networks when starting distccd
- **systemd service configuration**: Configure in service file

### Format Specification

The allowed networks format supports:
- **Single IP addresses**: `10.0.0.5`
- **CIDR notation**: `192.168.1.0/24` (allows entire subnet)
- **Multiple entries**: Separated by spaces or commas

### CIDR Range Examples

```
192.168.1.0/24          # Allows 192.168.1.1 through 192.168.1.254
10.0.0.0/8              # Allows 10.0.0.0 through 10.255.255.255
192.168.50.0/24         # Allows 192.168.50.1 through 192.168.50.254
```

### Multiple Network Example

```
10.0.0.5 192.168.1.0/24 172.16.0.0/16
```

This allows:
- Single host 10.0.0.5
- Entire 192.168.1.x subnet
- Entire 172.16.x.x network

### Default Behavior

**Important**: If DISTCC_ALLOW_NETS is not specified, distcc only accepts connections from localhost (127.0.0.1) for security.

### Configuration Methods

**Method 1: Environment Variable**
```
DISTCC_ALLOW_NETS="10.20.30.40 192.168.50.0/24"
```

**Method 2: Command Line**
```
distccd --daemon --allow 10.20.30.40 --allow 192.168.50.0/24
```

**Method 3: systemd Service File**
```
[Service]
Environment="DISTCC_ALLOW_NETS=10.20.30.40 192.168.50.0/24"
```

## 3. Job Distribution Best Practices

### Core Reservation

Always reserve some cores for system tasks and other processes:
- **Recommended**: Use 75% of available cores for distcc
- **Example**: On an 8-core system, use `/6` (6 jobs maximum)
- **Reason**: Prevents system overload and maintains responsiveness

### Local Compilation

Always include localhost in the host list:
- **Required**: Preprocessing typically happens locally
- **Performance**: Avoids network overhead for small files
- **Format**: List localhost first in configuration

```
localhost/4 remote1/6 remote2/6
```

### Optimal Job Distribution

Calculate job limits based on:
1. **Core count** on each node
2. **Reserve 25%** for system tasks
3. **Network capacity** between nodes

**Example Calculation**:
- 8-core node: 8 × 0.75 = 6 jobs
- 4-core node: 4 × 0.75 = 3 jobs
- Local 4-core: 4 jobs (use all cores)

### Load Balancing

distcc distributes jobs in the order hosts are listed:
- List fastest/most-capable hosts first
- Group hosts by network proximity
- Consider network latency

## 4. Configuration File Locations

### Client Configuration Files

**Per-user configuration**:
```
~/.distcc/hosts
```

**System-wide configuration**:
```
/etc/distcc/hosts
```

**Environment variable** (overrides file):
```
export DISTCC_HOSTS="localhost/4 192.168.1.10/8"
```

### Server Configuration

Servers are typically configured via:

**Command-line arguments**:
```bash
distccd --daemon --allow 192.168.1.0/24 --jobs 8
```

**systemd service file** (`/etc/systemd/system/distccd.service`):
```ini
[Unit]
Description=Distcc Compilation Server

[Service]
Type=forking
ExecStart=/usr/bin/distccd --daemon --allow 192.168.1.0/24
Environment="DISTCC_ALLOW_NETS=192.168.1.0/24"

[Install]
WantedBy=multi-user.target
```

**Documentation files**: Create `.conf` files for reference (not parsed by distcc):
```
/etc/distcc/server.conf
/tmp/distcc_configs/server.conf
```

## Complete Working Examples

### Example 1: Small Development Team

**Scenario**: 3 developers, 1 build server with 8 cores, 2 developer machines with 4 cores each

**Client Configuration** (`~/.distcc/hosts`):
```
localhost/4 buildserver.local/6 devmachine2.local/3
```

**Server Configuration** (on buildserver.local):
```
DISTCC_ALLOW_NETS="192.168.1.0/24"
```

### Example 2: CI/CD Pipeline

**Scenario**: Build coordinator distributing to dedicated build farm

**Client Configuration**:
```
localhost/2 build01.ci.local/8 build02.ci.local/8 build03.ci.local/4
```

**Server Configuration** (on build nodes):
```
DISTCC_ALLOW_NETS="10.0.50.100"
```
Only the CI coordinator (10.0.50.100) can submit jobs.

### Example 3: Mixed Environment with Security

**Scenario**: Production build server accepting from specific IPs and development subnet

**Client Configuration**:
```
localhost/4 node1.build.local/6 node2.build.local/6 node3.build.local/3
```

**Server Configuration** (on all build nodes):
```
DISTCC_ALLOW_NETS="10.20.30.40 192.168.50.0/24"
```

This allows:
- Build coordinator at 10.20.30.40
- All machines in development subnet 192.168.50.0/24

### Example 4: High-Performance Build Cluster

**Scenario**: Large C++ project, 4 powerful build servers

**Client Configuration**:
```
localhost/8 compile01/12 compile02/12 compile03/12 compile04/12
```

**Makefile integration**:
```makefile
export DISTCC_HOSTS=localhost/8 compile01/12 compile02/12 compile03/12 compile04/12
CXX=distcc g++
MAKEFLAGS=-j48
```

**Server Configuration**:
```
DISTCC_ALLOW_NETS="10.100.0.0/16"
```

This configuration:
- Uses 8 local jobs for preprocessing
- Distributes to 4 servers with 12 jobs each (48 remote jobs)
- Total parallelism: 56 concurrent compilation jobs
- Accepts connections from entire 10.100.x.x network

---

## Quick Reference Card

| Setting | Client | Server |
|---------|--------|--------|
| Environment Variable | DISTCC_HOSTS | DISTCC_ALLOW_NETS |
| Config File | ~/.distcc/hosts or /etc/distcc/hosts | N/A (use service config) |
| Format | `host/limit host/limit` | `IP CIDR IP CIDR` |
| Default Port | 3632 | 3632 |
| Example | `localhost/4 10.0.1.5/8` | `192.168.1.0/24` |