namespace py metrics

/**
 * ServerMetrics struct defines the performance metrics collected
 * from individual server instances during monitoring periods.
 * This schema is used for binary serialization of server performance data.
 */
struct ServerMetrics {
  /**
   * Unique identifier for the server instance
   */
  1: required i32 server_id,
  
  /**
   * Hostname or FQDN of the server
   */
  2: required string hostname,
  
  /**
   * Unix timestamp (seconds since epoch) when metrics were collected
   */
  3: required i64 timestamp,
  
  /**
   * CPU usage percentage at time of collection
   * Valid range: 0.0 to 100.0
   */
  4: required double cpu_percent,
  
  /**
   * Memory usage in bytes at time of collection
   */
  5: required i64 memory_bytes,
  
  /**
   * Number of errors logged during the monitoring period
   */
  6: required i32 error_count,
  
  /**
   * Current operational status of the server
   * Common values: 'active', 'idle', 'warning', 'critical'
   */
  7: required string status
}