-- Database Backup File
-- Generated: 2024-01-17 03:00:00
-- Database: production_db
-- Backup Type: Daily Full Backup
-- ================================================

-- Drop existing tables if they exist
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;

-- Create customers table
CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create products table
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2),
    stock_quantity INT DEFAULT 0
);

-- Create orders table
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    product_id INT,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    quantity INT,
    total_amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Insert sample customer data
INSERT INTO customers (customer_name, email) VALUES ('John Smith', 'john.smith@example.com');
INSERT INTO customers (customer_name, email) VALUES ('Jane Doe', 'jane.doe@example.com');
INSERT INTO customers (customer_name, email) VALUES ('Bob Johnson', 'bob.johnson@example.com');

-- Insert sample product data
INSERT INTO products (product_name, price, stock_quantity) VALUES ('Laptop', 999.99, 50);
INSERT INTO products (product_name, price, stock_quantity) VALUES ('Mouse', 29.99, 200);
INSERT INTO products (product_name, price, stock_quantity) VALUES ('Keyboard', 79.99, 150);

-- Insert sample order data
INSERT INTO orders (customer_id, product_id, quantity, total_amount) VALUES (1, 1, 1, 999.99);
INSERT INTO orders (customer_id, product_id, quantity, total_amount) VALUES (2, 2, 2, 59.98);
INSERT INTO orders (customer_id, product_id, quantity, total_amount) VALUES (3, 3, 1, 79.99);

-- End of backup file
-- Total tables backed up: 3
-- Backup completed successfully