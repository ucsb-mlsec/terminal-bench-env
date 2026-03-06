# S3 Write Requirements

## Critical Requirements

1. **Atomic Writes**: All writes to S3 must be atomic - either fully succeed or fully fail with no partial data
2. **No Orphaned Files**: Failed jobs must not leave temporary files or partial data in output locations
3. **Cleanup on Failure**: All staging and temporary files must be cleaned up after job failure
4. **Consistency**: No duplicate data or missing files after successful completion

## Current Issues

- Jobs leaving partial data in output location after failure
- Temporary files not being cleaned up in /tmp/s3a
- Duplicate records appearing after job retries
- Directory listings showing _temporary folders that never get removed

## Expected Behavior

- Output location should only contain data after successful job completion
- Failed jobs should leave output location unchanged
- No manual cleanup should be required
- Works with S3-compatible storage (MinIO, etc.)

## Performance Considerations

- Acceptable to trade some performance for consistency guarantees
- Staging overhead is acceptable if it ensures atomicity
- Jobs typically write 10-100GB per run

## Compatibility Requirements

- Must work with standard S3 API
- Must work with S3-compatible endpoints (MinIO, Ceph, etc.)
- Should support both Spark 2.x and 3.x where possible