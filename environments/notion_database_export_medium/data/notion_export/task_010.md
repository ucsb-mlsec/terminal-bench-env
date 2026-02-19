---
id: TASK-010
title: Database Migration to PostgreSQL 15
status: Blocked
assignee: Frank Brown
due_date: 2024-03-01
created_date: 2024-01-15
---

# Database Migration to PostgreSQL 15

## Description

Migrate the production database from PostgreSQL 13 to PostgreSQL 15 to take advantage of performance improvements and new features.

## Blocking Issues

- Waiting for infrastructure team to provision new database cluster
- Need approval from security team for configuration changes
- Compatibility testing revealed issues with legacy stored procedures

## Requirements

- Backup all existing data
- Test migration process in staging environment
- Update connection strings across all services
- Validate data integrity post-migration
- Document rollback procedures

## Dependencies

- Infrastructure team capacity
- Security review completion
- Development team availability for testing

## Notes

Migration has been delayed due to resource constraints in the infrastructure team. Expected to unblock by mid-March pending infrastructure allocation.