#!/bin/bash

# Setup comprehensive audit test environment
# Creates a complex directory structure with varying permissions

set -e

BASE_DIR="/var/audit_test"

# Clean up if exists
rm -rf "$BASE_DIR"

# Create base directory
mkdir -p "$BASE_DIR"

echo "Creating comprehensive test environment at $BASE_DIR..."

# 1. PUBLIC SECTION - Accessible area
mkdir -p "$BASE_DIR/public/data"
mkdir -p "$BASE_DIR/public/logs/archive"
mkdir -p "$BASE_DIR/public/config"
mkdir -p "$BASE_DIR/public/reports/monthly/2024"
mkdir -p "$BASE_DIR/public/reports/weekly"

touch "$BASE_DIR/public/readme.txt"
touch "$BASE_DIR/public/data/users.csv"
touch "$BASE_DIR/public/data/products.json"
touch "$BASE_DIR/public/logs/access.log"
touch "$BASE_DIR/public/logs/error.log"
touch "$BASE_DIR/public/logs/archive/old.log"
touch "$BASE_DIR/public/config/settings.conf"
touch "$BASE_DIR/public/config/database.ini"
touch "$BASE_DIR/public/reports/summary.txt"
touch "$BASE_DIR/public/reports/monthly/2024/jan.txt"
touch "$BASE_DIR/public/reports/weekly/week1.txt"

chmod 755 "$BASE_DIR/public"
chmod 755 "$BASE_DIR/public/data"
chmod 755 "$BASE_DIR/public/logs"
chmod 755 "$BASE_DIR/public/logs/archive"
chmod 755 "$BASE_DIR/public/config"
chmod 755 "$BASE_DIR/public/reports"
chmod 755 "$BASE_DIR/public/reports/monthly"
chmod 755 "$BASE_DIR/public/reports/monthly/2024"
chmod 755 "$BASE_DIR/public/reports/weekly"
chmod 644 "$BASE_DIR/public"/*.txt 2>/dev/null || true
chmod 644 "$BASE_DIR/public/data"/*
chmod 644 "$BASE_DIR/public/logs"/*.log
chmod 644 "$BASE_DIR/public/logs/archive"/*
chmod 644 "$BASE_DIR/public/config"/*
chmod 644 "$BASE_DIR/public/reports"/*.txt
chmod 644 "$BASE_DIR/public/reports/monthly/2024"/*
chmod 644 "$BASE_DIR/public/reports/weekly"/*

# 2. RESTRICTED SECTION - Owner only
mkdir -p "$BASE_DIR/restricted/admin/credentials"
mkdir -p "$BASE_DIR/restricted/admin/keys"
mkdir -p "$BASE_DIR/restricted/private/user_data"
mkdir -p "$BASE_DIR/restricted/private/backups"

touch "$BASE_DIR/restricted/admin/admin.conf"
touch "$BASE_DIR/restricted/admin/credentials/db_password.txt"
touch "$BASE_DIR/restricted/admin/credentials/api_keys.txt"
touch "$BASE_DIR/restricted/admin/keys/private.key"
touch "$BASE_DIR/restricted/admin/keys/public.key"
touch "$BASE_DIR/restricted/private/secrets.txt"
touch "$BASE_DIR/restricted/private/user_data/sensitive.dat"
touch "$BASE_DIR/restricted/private/backups/backup.tar.gz"

chmod 700 "$BASE_DIR/restricted"
chmod 700 "$BASE_DIR/restricted/admin"
chmod 700 "$BASE_DIR/restricted/admin/credentials"
chmod 700 "$BASE_DIR/restricted/admin/keys"
chmod 700 "$BASE_DIR/restricted/private"
chmod 700 "$BASE_DIR/restricted/private/user_data"
chmod 700 "$BASE_DIR/restricted/private/backups"
chmod 600 "$BASE_DIR/restricted/admin"/*.conf 2>/dev/null || true
chmod 600 "$BASE_DIR/restricted/admin/credentials"/*
chmod 600 "$BASE_DIR/restricted/admin/keys"/*
chmod 600 "$BASE_DIR/restricted/private"/*.txt 2>/dev/null || true
chmod 600 "$BASE_DIR/restricted/private/user_data"/*
chmod 600 "$BASE_DIR/restricted/private/backups"/*

# 3. DENIED SECTION - No access at all
mkdir -p "$BASE_DIR/denied/vault/level1/level2"
mkdir -p "$BASE_DIR/denied/quarantine"

touch "$BASE_DIR/denied/blocked.txt"
touch "$BASE_DIR/denied/vault/secure.dat"
touch "$BASE_DIR/denied/vault/level1/deep.txt"
touch "$BASE_DIR/denied/vault/level1/level2/deeper.txt"
touch "$BASE_DIR/denied/quarantine/suspicious.bin"

chmod 000 "$BASE_DIR/denied"
chmod 000 "$BASE_DIR/denied/vault"
chmod 000 "$BASE_DIR/denied/vault/level1"
chmod 000 "$BASE_DIR/denied/vault/level1/level2"
chmod 000 "$BASE_DIR/denied/quarantine"
chmod 000 "$BASE_DIR/denied"/*.txt 2>/dev/null || true
chmod 000 "$BASE_DIR/denied/vault"/*.dat 2>/dev/null || true
chmod 000 "$BASE_DIR/denied/vault/level1"/*.txt 2>/dev/null || true
chmod 000 "$BASE_DIR/denied/vault/level1/level2"/*.txt 2>/dev/null || true
chmod 000 "$BASE_DIR/denied/quarantine"/*

# 4. MIXED SECTION - Complex permissions
mkdir -p "$BASE_DIR/mixed/open/subdir1/subdir2"
mkdir -p "$BASE_DIR/mixed/closed/hidden"
mkdir -p "$BASE_DIR/mixed/partial/accessible"
mkdir -p "$BASE_DIR/mixed/partial/blocked"

touch "$BASE_DIR/mixed/index.txt"
touch "$BASE_DIR/mixed/open/public.txt"
touch "$BASE_DIR/mixed/open/subdir1/file1.txt"
touch "$BASE_DIR/mixed/open/subdir1/subdir2/file2.txt"
touch "$BASE_DIR/mixed/closed/locked.txt"
touch "$BASE_DIR/mixed/closed/hidden/secret.txt"
touch "$BASE_DIR/mixed/partial/readme.txt"
touch "$BASE_DIR/mixed/partial/accessible/open.txt"
touch "$BASE_DIR/mixed/partial/blocked/forbidden.txt"

chmod 755 "$BASE_DIR/mixed"
chmod 755 "$BASE_DIR/mixed/open"
chmod 755 "$BASE_DIR/mixed/open/subdir1"
chmod 755 "$BASE_DIR/mixed/open/subdir2" 2>/dev/null || true
chmod 000 "$BASE_DIR/mixed/closed"
chmod 000 "$BASE_DIR/mixed/closed/hidden"
chmod 755 "$BASE_DIR/mixed/partial"
chmod 755 "$BASE_DIR/mixed/partial/accessible"
chmod 000 "$BASE_DIR/mixed/partial/blocked"
chmod 644 "$BASE_DIR/mixed"/*.txt 2>/dev/null || true
chmod 644 "$BASE_DIR/mixed/open"/*.txt 2>/dev/null || true
chmod 644 "$BASE_DIR/mixed/open/subdir1"/*.txt 2>/dev/null || true
chmod 644 "$BASE_DIR/mixed/open/subdir1/subdir2"/*.txt 2>/dev/null || true
chmod 000 "$BASE_DIR/mixed/closed"/*.txt 2>/dev/null || true
chmod 000 "$BASE_DIR/mixed/closed/hidden"/*.txt 2>/dev/null || true
chmod 644 "$BASE_DIR/mixed/partial"/*.txt 2>/dev/null || true
chmod 644 "$BASE_DIR/mixed/partial/accessible"/*.txt 2>/dev/null || true
chmod 000 "$BASE_DIR/mixed/partial/blocked"/*.txt 2>/dev/null || true

# 5. APPLICATIONS SECTION - Multi-level structure
mkdir -p "$BASE_DIR/applications/web/frontend/assets/css"
mkdir -p "$BASE_DIR/applications/web/frontend/assets/js"
mkdir -p "$BASE_DIR/applications/web/backend/api"
mkdir -p "$BASE_DIR/applications/web/backend/database"
mkdir -p "$BASE_DIR/applications/services/auth"
mkdir -p "$BASE_DIR/applications/services/payment"

touch "$BASE_DIR/applications/README.md"
touch "$BASE_DIR/applications/web/index.html"
touch "$BASE_DIR/applications/web/frontend/app.js"
touch "$BASE_DIR/applications/web/frontend/assets/css/style.css"
touch "$BASE_DIR/applications/web/frontend/assets/js/main.js"
touch "$BASE_DIR/applications/web/backend/server.js"
touch "$BASE_DIR/applications/web/backend/api/routes.js"
touch "$BASE_DIR/applications/web/backend/database/schema.sql"
touch "$BASE_DIR/applications/services/auth/auth.conf"
touch "$BASE_DIR/applications/services/payment/payment.conf"

chmod 755 "$BASE_DIR/applications"
chmod 755 "$BASE_DIR/applications/web"
chmod 755 "$BASE_DIR/applications/web/frontend"
chmod 755 "$BASE_DIR/applications/web/frontend/assets"
chmod 755 "$BASE_DIR/applications/web/frontend/assets/css"
chmod 755 "$BASE_DIR/applications/web/frontend/assets/js"
chmod 755 "$BASE_DIR/applications/web/backend"
chmod 755 "$BASE_DIR/applications/web/backend/api"
chmod 700 "$BASE_DIR/applications/web/backend/database"
chmod 755 "$BASE_DIR/applications/services"
chmod 700 "$BASE_DIR/applications/services/auth"
chmod 700 "$BASE_DIR/applications/services/payment"
chmod 644 "$BASE_DIR/applications"/*.md 2>/dev/null || true
chmod 644 "$BASE_DIR/applications/web"/*.html 2>/dev/null || true
chmod 644 "$BASE_DIR/applications/web/frontend"/*.js 2>/dev/null || true
chmod 644 "$BASE_DIR/applications/web/frontend/assets/css"/*
chmod 644 "$BASE_DIR/applications/web/frontend/assets/js"/*
chmod 644 "$BASE_DIR/applications/web/backend"/*.js 2>/dev/null || true
chmod 644 "$BASE_DIR/applications/web/backend/api"/*
chmod 600 "$BASE_DIR/applications/web/backend/database"/* 2>/dev/null || true
chmod 600 "$BASE_DIR/applications/services/auth"/* 2>/dev/null || true
chmod 600 "$BASE_DIR/applications/services/payment"/* 2>/dev/null || true

# 6. TENANTS SECTION - Multi-tenant structure
mkdir -p "$BASE_DIR/tenants/tenant1/data"
mkdir -p "$BASE_DIR/tenants/tenant1/logs"
mkdir -p "$BASE_DIR/tenants/tenant2/data"
mkdir -p "$BASE_DIR/tenants/tenant2/logs"
mkdir -p "$BASE_DIR/tenants/tenant3/data"
mkdir -p "$BASE_DIR/tenants/tenant3/logs"

touch "$BASE_DIR/tenants/tenant1/config.json"
touch "$BASE_DIR/tenants/tenant1/data/records.db"
touch "$BASE_DIR/tenants/tenant1/logs/activity.log"
touch "$BASE_DIR/tenants/tenant2/config.json"
touch "$BASE_DIR/tenants/tenant2/data/records.db"
touch "$BASE_DIR/tenants/tenant2/logs/activity.log"
touch "$BASE_DIR/tenants/tenant3/config.json"
touch "$BASE_DIR/tenants/tenant3/data/records.db"
touch "$BASE_DIR/tenants/tenant3/logs/activity.log"

chmod 755 "$BASE_DIR/tenants"
chmod 755 "$BASE_DIR/tenants/tenant1"
chmod 755 "$BASE_DIR/tenants/tenant1/data"
chmod 755 "$BASE_DIR/tenants/tenant1/logs"
chmod 700 "$BASE_DIR/tenants/tenant2"
chmod 700 "$BASE_DIR/tenants/tenant2/data"
chmod 700 "$BASE_DIR/tenants/tenant2/logs"
chmod 000 "$BASE_DIR/tenants/tenant3"
chmod 000 "$BASE_DIR/tenants/tenant3/data"
chmod 000 "$BASE_DIR/tenants/tenant3/logs"
chmod 644 "$BASE_DIR/tenants/tenant1"/*.json 2>/dev/null || true
chmod 644 "$BASE_DIR/tenants/tenant1/data"/*
chmod 644 "$BASE_DIR/tenants/tenant1/logs"/*
chmod 600 "$BASE_DIR/tenants/tenant2"/*.json 2>/dev/null || true
chmod 600 "$BASE_DIR/tenants/tenant2/data"/* 2>/dev/null || true
chmod 600 "$BASE_DIR/tenants/tenant2/logs"/* 2>/dev/null || true
chmod 000 "$BASE_DIR/tenants/tenant3"/*.json 2>/dev/null || true
chmod 000 "$BASE_DIR/tenants/tenant3/data"/* 2>/dev/null || true
chmod 000 "$BASE_DIR/tenants/tenant3/logs"/* 2>/dev/null || true

# 7. SYMLINKS SECTION
mkdir -p "$BASE_DIR/links"
chmod 755 "$BASE_DIR/links"

ln -s "$BASE_DIR/public" "$BASE_DIR/links/link_to_public"
ln -s "$BASE_DIR/restricted" "$BASE_DIR/links/link_to_restricted"
ln -s "$BASE_DIR/denied" "$BASE_DIR/links/link_to_denied"
ln -s "$BASE_DIR/public/data/users.csv" "$BASE_DIR/links/users_link"
ln -s "/nonexistent/path" "$BASE_DIR/links/broken_link"

# 8. SYSTEM SECTION - Simulated system files
mkdir -p "$BASE_DIR/system/etc/config.d"
mkdir -p "$BASE_DIR/system/var/lib/data"
mkdir -p "$BASE_DIR/system/var/cache"
mkdir -p "$BASE_DIR/system/var/tmp"

touch "$BASE_DIR/system/etc/system.conf"
touch "$BASE_DIR/system/etc/config.d/app1.conf"
touch "$BASE_DIR/system/etc/config.d/app2.conf"
touch "$BASE_DIR/system/var/lib/data/state.db"
touch "$BASE_DIR/system/var/cache/cache.dat"
touch "$BASE_DIR/system/var/tmp/temp.txt"

chmod 755 "$BASE_DIR/system"
chmod 755 "$BASE_DIR/system/etc"
chmod 755 "$BASE_DIR/system/etc/config.d"
chmod 755 "$BASE_DIR/system/var"
chmod 755 "$BASE_DIR/system/var/lib"
chmod 755 "$BASE_DIR/system/var/lib/data"
chmod 000 "$BASE_DIR/system/var/cache"
chmod 755 "$BASE_DIR/system/var/tmp"
chmod 644 "$BASE_DIR/system/etc"/*.conf 2>/dev/null || true
chmod 644 "$BASE_DIR/system/etc/config.d"/*
chmod 644 "$BASE_DIR/system/var/lib/data"/*
chmod 000 "$BASE_DIR/system/var/cache"/* 2>/dev/null || true
chmod 644 "$BASE_DIR/system/var/tmp"/*

# 9. EMPTY DIRECTORIES
mkdir -p "$BASE_DIR/empty/dir1"
mkdir -p "$BASE_DIR/empty/dir2"
mkdir -p "$BASE_DIR/empty/restricted_empty"

chmod 755 "$BASE_DIR/empty"
chmod 755 "$BASE_DIR/empty/dir1"
chmod 755 "$BASE_DIR/empty/dir2"
chmod 000 "$BASE_DIR/empty/restricted_empty"

echo "Test environment created successfully!"
echo "Structure summary:"
find "$BASE_DIR" -type d 2>/dev/null | wc -l | xargs echo "  Directories created:"
find "$BASE_DIR" -type f 2>/dev/null | wc -l | xargs echo "  Files created:"
find "$BASE_DIR" -type l 2>/dev/null | wc -l | xargs echo "  Symlinks created:"
