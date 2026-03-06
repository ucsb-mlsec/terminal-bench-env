HDFS Datanode Configuration Guide
==========================================

1. DFS.DATANODE.DATA.DIR PROPERTY
----------------------------------
Purpose:
  - Specifies the storage volumes where HDFS will store data blocks
  - Each path represents a separate physical storage volume
  - Datanode distributes blocks across all configured volumes

Format:
  - Comma-separated list of directory paths
  - NO SPACES between paths
  - Example: /data/volume1,/data/volume2,/data/volume3

2. VOLUME FAILURE HANDLING
---------------------------
Critical Guidelines:
  - Failed volumes MUST be removed from the active configuration
  - Including failed volumes will prevent datanode startup
  - Only OPERATIONAL volumes should be listed in dfs.datanode.data.dir

Related Property:
  - dfs.datanode.failed.volumes.tolerated
  - Specifies how many volume failures the datanode can tolerate
  - Default is 0 (any failure stops the datanode)
  - Set higher for graceful degradation

Best Practices:
  - Regularly monitor volume_status.log for failures
  - Remove failed volumes immediately from configuration
  - Replace failed hardware before re-adding volumes
  - Maintain at least 2 healthy volumes for redundancy

3. CONFIGURATION VALIDATION
----------------------------
Identifying Failed Volumes:
  - Check volume_status.log for status indicators
  - Look for keywords: FAILED, UNAVAILABLE, ERROR
  - OPERATIONAL status indicates healthy volumes

Update Steps:
  1. Parse volume_status.log to identify healthy volumes
  2. Extract paths of OPERATIONAL volumes only
  3. Create comma-separated list (no spaces)
  4. Update dfs.datanode.data.dir with healthy volumes only

WARNING: Including failed volumes in configuration will cause datanode failure!