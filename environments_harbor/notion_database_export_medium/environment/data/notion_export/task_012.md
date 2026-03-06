---
id: TASK-012
title: Database migration to PostgreSQL 15
status: Blocked
assignee: Bob Wilson
due_date: 2024-11-30
created_date: 2024-10-15
---

# Database migration to PostgreSQL 15

## Description

Migrate the production database from PostgreSQL 13 to PostgreSQL 15 to take advantage of performance improvements and new features.

## Blocking Issues

- Waiting for infrastructure team to provision new database servers
- Need approval from security team for the migration plan
- Third-party reporting tool compatibility needs to be verified

## Tasks

- [ ] Review PostgreSQL 15 breaking changes
- [ ] Test application compatibility with PostgreSQL 15
- [ ] Create detailed migration runbook
- [ ] Schedule maintenance window
- [ ] Coordinate with DevOps team

## Notes

The migration has been delayed due to resource constraints in the infrastructure team. Target completion is end of November, but this may slip into December depending on server availability.

## Dependencies

- Server provisioning (Infrastructure team)
- Security approval
- Vendor compatibility confirmation