#!/bin/bash

# Optimized query execution script
# This script reorganizes the data for efficient querying and executes the optimized query

SOLUTION_DIR="/solution"
DATA_DIR="${SOLUTION_DIR}/data"
RESULTS_FILE="${SOLUTION_DIR}/results.json"
ORIGINAL_DATA="/data/orders"

# Create solution directory structure
mkdir -p "${DATA_DIR}"

# Check if optimized data already exists
if [ ! -f "${DATA_DIR}/orders_2023_q4_electronics.parquet" ]; then
    echo "Creating optimized data structure..."
    
    # Reorganize data by:
    # 1. Filtering to only 2023 Q4 data
    # 2. Further partitioning by category to enable category-based filtering
    # 3. Creating separate optimized files for each category
    
    /usr/bin/duckdb :memory: <<SQL
    -- Create optimized partition for Electronics in Q4 2023
    COPY (
        SELECT *
        FROM read_parquet('${ORIGINAL_DATA}/*/*.parquet')
        WHERE order_date >= '2023-10-01' 
          AND order_date < '2024-01-01'
          AND category = 'Electronics'
    ) TO '${DATA_DIR}/orders_2023_q4_electronics.parquet' (FORMAT PARQUET);
    
    -- Create optimized partition for other Q4 2023 data (for data preservation)
    COPY (
        SELECT *
        FROM read_parquet('${ORIGINAL_DATA}/*/*.parquet')
        WHERE order_date >= '2023-10-01' 
          AND order_date < '2024-01-01'
          AND category != 'Electronics'
    ) TO '${DATA_DIR}/orders_2023_q4_other.parquet' (FORMAT PARQUET);
    
    -- Preserve all other 2023 data (non-Q4)
    COPY (
        SELECT *
        FROM read_parquet('${ORIGINAL_DATA}/*/*.parquet')
        WHERE order_date >= '2023-01-01' 
          AND order_date < '2023-10-01'
    ) TO '${DATA_DIR}/orders_2023_other.parquet' (FORMAT PARQUET);
    
    -- Preserve all 2022 data
    COPY (
        SELECT *
        FROM read_parquet('${ORIGINAL_DATA}/*/*.parquet')
        WHERE order_date < '2023-01-01'
    ) TO '${DATA_DIR}/orders_2022.parquet' (FORMAT PARQUET);
SQL
    
    echo "Optimized data structure created."
else
    echo "Optimized data structure already exists."
fi

# Execute the optimized query
# Now we only need to scan the specific file containing Q4 2023 Electronics data
/usr/bin/duckdb :memory: <<SQL > "${RESULTS_FILE}"
SELECT 
    region,
    SUM(amount) as total_revenue
FROM read_parquet('${DATA_DIR}/orders_2023_q4_electronics.parquet')
GROUP BY region
ORDER BY region;
SQL

# Convert output to proper JSON format
# DuckDB's default output needs to be converted to JSON array format
/usr/bin/duckdb :memory: <<SQL > "${RESULTS_FILE}"
COPY (
    SELECT 
        region,
        SUM(amount) as total_revenue
    FROM read_parquet('${DATA_DIR}/orders_2023_q4_electronics.parquet')
    GROUP BY region
    ORDER BY region
) TO '/dev/stdout' (FORMAT JSON, ARRAY true);
SQL

echo "Query executed. Results saved to ${RESULTS_FILE}"