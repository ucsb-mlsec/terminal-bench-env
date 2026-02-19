-- Database Backup File
-- Generated: 2024-01-15 03:00:00
-- Database: production_db
-- Type: Daily Backup
-- ============================================

-- Drop existing tables if they exist
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;

-- Create users table
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert user data
INSERT INTO users (id, name, email) VALUES (1, 'John Doe', 'john.doe@example.com');
INSERT INTO users (id, name, email) VALUES (2, 'Jane Smith', 'jane.smith@example.com');
INSERT INTO users (id, name, email) VALUES (3, 'Bob Johnson', 'bob.j@example.com');

-- Create products table
CREATE TABLE products (
    id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100),
    price DECIMAL(10,2),
    stock_quantity INT
);

-- Insert product data
INSERT INTO products (id, product_name, price, stock_quantity) VALUES (1, 'Laptop', 999.99, 50);
INSERT INTO products (id, product_name, price, stock_quantity) VALUES (2, 'Mouse', 29.99, 200);
INSERT INTO products (id, product_name, price, stock_quantity) VALUES (3, 'Keyboard', 79.99, 150);

-- End of backup file