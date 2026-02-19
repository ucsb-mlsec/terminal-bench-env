#!/bin/bash
set -e

DB_PATH="/home/data/analytics.db"

echo "Refreshing materialized views..."

# Wrong order - refreshing combined_summary first, which depends on the others
echo "Refreshing mv_combined_summary..."
sqlite3 "$DB_PATH" "DELETE FROM mv_combined_summary;"
sqlite3 "$DB_PATH" "INSERT INTO mv_combined_summary SELECT p.category, r.region_name, SUM(s.amount) as total_sales, COUNT(*) as sale_count FROM sales s JOIN products p ON s.product_id = p.product_id JOIN regions r ON s.region_id = r.region_id GROUP BY p.category, r.region_name;"

echo "Refreshing mv_region_totals..."
sqlite3 "$DB_PATH" "DELETE FROM mv_region_totals;"
sqlite3 "$DB_PATH" "INSERT INTO mv_region_totals SELECT r.region_id, r.region_name, SUM(s.amount) as total_sales, COUNT(*) as sale_count FROM sales s JOIN regions r ON s.region_id = r.region_id GROUP BY r.region_id, r.region_name;"

echo "Refreshing mv_product_totals..."
sqlite3 "$DB_PATH" "DELETE FROM mv_product_totals;"
sqlite3 "$DB_PATH" "INSERT INTO mv_product_totals SELECT p.product_id, p.product_name, p.category, SUM(s.amount) as total_sales, COUNT(*) as sale_count FROM sales s JOIN products p ON s.product_id = p.product_id GROUP BY p.product_id, p.product_name, p.category;"

echo "Refresh complete"