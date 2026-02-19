#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sentry Self-Hosted Configuration File
This file contains the main configuration for the Sentry instance.
"""

import os

# System Configuration
SENTRY_SYSTEM_MAX_EVENTS_PER_MINUTE = 5000
SENTRY_SINGLE_ORGANIZATION = True

# Basic Sentry Settings
SENTRY_OPTIONS = {
    'system.url-prefix': os.getenv('SENTRY_URL_PREFIX', 'http://localhost:9000'),
    'system.secret-key': os.getenv('SENTRY_SECRET_KEY', 'changeme'),
    'redis.clusters': {
        'default': {
            'hosts': {
                0: {
                    'host': 'redis',
                    'port': 6379,
                    'db': 0,
                }
            }
        }
    },
}

# Database Configuration
DATABASES = {
    'default': {
        'ENGINE': 'sentry.db.postgres',
        'NAME': 'sentry',
        'USER': 'sentry',
        'PASSWORD': os.getenv('SENTRY_DB_PASSWORD', 'sentry'),
        'HOST': 'postgres',
        'PORT': '5432',
    }
}

# Rate Limiting Configuration
RATELIMIT_ENABLED = True

# Global rate limiting settings
SENTRY_RATELIMITER = 'sentry.ratelimits.redis.RedisRateLimiter'
SENTRY_RATELIMITER_OPTIONS = {}

# Per-project rate limits - NEEDS CONFIGURATION
# TODO: Configure per-project rate limits properly
# Attempt 1: Load from external file (not working?)
# try:
#     import json
#     with open('/etc/sentry/rate_limits.json', 'r') as f:
#         PROJECT_RATE_LIMITS = json.load(f)
# except:
#     PROJECT_RATE_LIMITS = {}

# Attempt 2: Direct configuration (incomplete)
# SENTRY_PROJECT_RATE_LIMIT = {
#     'default': 100,
#     # Add per-project limits here?
# }

# Attempt 3: Using quotas? (not sure if this is right)
# SENTRY_QUOTAS = 'sentry.quotas.redis.RedisQuota'
# SENTRY_QUOTA_OPTIONS = {}

# NOTE: Multiple projects are currently unprotected by rate limits
# High-volume projects are overwhelming the system
# Critical projects need priority handling
# Current state: INCOMPLETE AND MISCONFIGURED
