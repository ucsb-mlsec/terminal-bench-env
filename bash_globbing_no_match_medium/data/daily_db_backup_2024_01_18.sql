-- Database Backup File
-- Generated: 2024-01-18
-- Database: production_db
-- Backup Type: Daily Full Backup
-- Server: db-prod-01

-- Drop existing tables if they exist
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;

-- Create users table
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert user data
INSERT INTO users (username, email) VALUES ('john_doe', 'john@example.com');
INSERT INTO users (username, email) VALUES ('jane_smith', 'jane@example.com');
INSERT INTO users (username, email) VALUES ('bob_wilson', 'bob@example.com');

-- Create products table
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    stock_quantity INT DEFAULT 0
);

-- Insert product data
INSERT INTO products (product_name, price, stock_quantity) VALUES ('Laptop', 999.99, 50);
INSERT INTO products (product_name, price, stock_quantity) VALUES ('Mouse', 29.99, 200);
INSERT INTO products (product_name, price, stock_quantity) VALUES ('Keyboard', 79.99, 150);

-- Create orders table
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    product_id INT NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Insert order data
INSERT INTO orders (user_id, product_id) VALUES (1, 1);
INSERT INTO orders (user_id, product_id) VALUES (2, 2);
INSERT INTO orders (user_id, product_id) VALUES (3, 3);

-- End of backup file
-- Total tables backed up: 3
-- Backup completed successfully