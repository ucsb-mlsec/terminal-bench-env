=== Customer Transaction Processor ===

This application processes daily transaction batches.

Setup Notes:
- Configuration files are in multiple locations
- Some team members edited spark-defaults.conf
- Others modified application.properties
- Nobody remembers which settings are actually active

Performance Issues:
- Inconsistent runtime (sometimes 2 hours, sometimes 6 hours)
- Memory usage varies significantly
- Batch processing seems slow

Recent Changes:
- Jake modified something in spark-defaults.conf last month
- Sarah added arrow settings somewhere (maybe application.properties?)
- Default batch size might be overridden, not sure where
- Vectorized reader was enabled by someone, then maybe disabled?

Configuration File Locations (we think):
- /opt/spark-app/conf/spark-defaults.conf
- /opt/spark-app/config/application.properties
- Maybe something in /opt/spark-app/settings/ ?

TODO:
- Consolidate configuration files
- Document active settings
- Optimize columnar batch processing
- Check vectorization settings
- Figure out what batch size we're actually using

URGENT: Need to identify current batch processing config before next optimization sprint!

Last Updated: Multiple team members, dates unknown