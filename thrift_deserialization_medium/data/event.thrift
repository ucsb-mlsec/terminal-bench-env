namespace py monitoring
namespace java com.monitoring
namespace cpp monitoring

/**
 * Severity levels for monitoring events.
 * Events with ERROR or CRITICAL severity indicate problems that need investigation.
 */
enum Severity {
  INFO = 1,
  WARNING = 2,
  ERROR = 3,
  CRITICAL = 4
}

/**
 * Status of an event or operation.
 * Events with FAILURE status indicate problems that need investigation.
 */
enum Status {
  SUCCESS = 1,
  FAILURE = 2,
  PENDING = 3
}

/**
 * Event structure for monitoring system.
 * 
 * Problem Detection Rules:
 * - Events with severity ERROR or CRITICAL require investigation
 * - Events with status FAILURE require investigation
 * - An event is considered a problem if either condition is met
 */
struct Event {
  1: required string eventId,
  2: required i64 timestamp,
  3: required Severity severity,
  4: required Status status,
  5: required string serviceName,
  6: optional string message
}