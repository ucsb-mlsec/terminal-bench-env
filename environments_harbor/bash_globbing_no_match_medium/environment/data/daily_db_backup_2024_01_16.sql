-- Database Backup File
-- Generated: 2024-01-16 03:00:00
-- Database: production_db
-- Server: db-prod-01
-- Backup Type: Daily Full Backup

-- Drop tables if they exist
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS orders;

-- Create products table
CREATE TABLE products (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2),
    stock_quantity INT DEFAULT 0
);

-- Create customers table
CREATE TABLE customers (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create orders table
CREATE TABLE orders (
    id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    product_id INT,
    quantity INT,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- Insert sample data into products
INSERT INTO products (name, price, stock_quantity) VALUES ('Laptop', 999.99, 50);
INSERT INTO products (name, price, stock_quantity) VALUES ('Mouse', 29.99, 200);
INSERT INTO products (name, price, stock_quantity) VALUES ('Keyboard', 79.99, 150);
INSERT INTO products (name, price, stock_quantity) VALUES ('Monitor', 299.99, 75);

-- Insert sample data into customers
INSERT INTO customers (name, email) VALUES ('John Smith', 'john.smith@example.com');
INSERT INTO customers (name, email) VALUES ('Jane Doe', 'jane.doe@example.com');
INSERT INTO customers (name, email) VALUES ('Bob Wilson', 'bob.wilson@example.com');

-- Insert sample data into orders
INSERT INTO orders (customer_id, product_id, quantity) VALUES (1, 1, 1);
INSERT INTO orders (customer_id, product_id, quantity) VALUES (2, 2, 2);
INSERT INTO orders (customer_id, product_id, quantity) VALUES (1, 3, 1);
INSERT INTO orders (customer_id, product_id, quantity) VALUES (3, 4, 1);

-- Backup completed successfully
-- End of backup file