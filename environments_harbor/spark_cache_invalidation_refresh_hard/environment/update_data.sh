#!/bin/bash

echo "Updating transaction data..."

# Append new transaction rows to transactions.csv
echo "T501,C101,250.00,2024-01-15,Electronics" >> /workspace/data/transactions.csv
echo "T502,C102,175.50,2024-01-15,Clothing" >> /workspace/data/transactions.csv
echo "T503,C103,89.99,2024-01-16,Food" >> /workspace/data/transactions.csv
echo "T504,C104,450.00,2024-01-16,Electronics" >> /workspace/data/transactions.csv
echo "T505,C105,125.75,2024-01-17,Books" >> /workspace/data/transactions.csv
echo "T506,C501,300.00,2024-01-17,Electronics" >> /workspace/data/transactions.csv
echo "T507,C502,199.99,2024-01-18,Clothing" >> /workspace/data/transactions.csv
echo "T508,C101,75.00,2024-01-18,Food" >> /workspace/data/transactions.csv

echo "Updating customer data..."

# Append new customer rows to customers.csv
echo "C501,Alice Johnson,alice.johnson@email.com,Seattle,Gold" >> /workspace/data/customers.csv
echo "C502,Bob Williams,bob.williams@email.com,Portland,Silver" >> /workspace/data/customers.csv
echo "C503,Carol Martinez,carol.martinez@email.com,Denver,Platinum" >> /workspace/data/customers.csv

# Modify an existing transaction amount (optional - change first data row amount)
sed -i 's/T101,C101,[0-9.]*,/T101,C101,999.99,/' /workspace/data/transactions.csv

echo "Data update complete"