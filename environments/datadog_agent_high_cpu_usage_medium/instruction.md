A Datadog monitoring agent has been deployed on this production server to collect system metrics and application performance data. The agent was functioning normally with minimal resource usage until this morning, when it suddenly began consuming 80-95% CPU continuously. This high CPU usage is degrading performance for other critical services running on the server.

**Current Situation:**
- The Datadog agent process is running and consuming excessive CPU
- Normal CPU usage for the agent should be under 5%
- The agent is still collecting metrics, but the performance impact is unacceptable
- System logs show the agent has been running in this state for approximately 3 hours
- The agent's configuration file is located at `/etc/datadog-agent/datadog.yaml`

**Background:**
The Datadog agent was recently reconfigured by a junior team member who made several changes to the configuration file. The high CPU usage started immediately after these configuration changes were applied. The previous configuration file is not available, so you'll need to identify and fix the problematic settings by investigating the current configuration and the agent's behavior.

**Your Task:**
You need to identify what's causing the excessive CPU consumption and fix it. The issue is configuration-related - no code changes are needed to the agent itself. Once you've identified and corrected the problem, the agent should return to normal CPU usage levels (under 5%).

**Investigation Resources:**
- The agent's configuration file at `/etc/datadog-agent/datadog.yaml`
- Agent logs available in `/var/log/datadog/`
- The running agent process (you can observe its behavior)
- Agent status and diagnostic commands may be available

**Output Requirements:**
Save your solution to `/root/solution.txt` as a plain text file with exactly 3 lines:

Line 1: The configuration parameter that was causing the high CPU usage
Line 2: The problematic value that was set
Line 3: The corrected value you applied

Example format:
```
check_interval
1
60
```

**Success Criteria:**
- The solution.txt file exists at `/root/solution.txt`
- The file contains exactly 3 lines (parameter name, old value, new value)
- Each line contains only the value (no labels, no extra text)
- The configuration file has been corrected with your fix
- The fix addresses the root cause of the CPU issue

Note: Focus on finding the configuration problem through investigation of the agent's current state, logs, and configuration. The agent binary and core functionality are working correctly - this is purely a configuration issue.
