-- Monthly Database Backup - December
-- Generated: 2024-12-01
-- Database: production_db
-- Backup Type: Full Monthly Backup
-- =============================================

-- Drop existing tables if they exist
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;

-- Create users table
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active'
);

-- Insert sample user data
INSERT INTO users (username, email, status) VALUES
('john_doe', 'john.doe@example.com', 'active'),
('jane_smith', 'jane.smith@example.com', 'active'),
('bob_wilson', 'bob.wilson@example.com', 'inactive'),
('alice_brown', 'alice.brown@example.com', 'active'),
('charlie_davis', 'charlie.davis@example.com', 'active');

-- Create products table
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    price DECIMAL(10, 2),
    stock_quantity INT DEFAULT 0
);

-- Insert sample product data
INSERT INTO products (product_name, category, price, stock_quantity) VALUES
('Laptop Computer', 'Electronics', 1299.99, 45),
('Wireless Mouse', 'Electronics', 29.99, 150),
('Office Chair', 'Furniture', 249.99, 30),
('Desk Lamp', 'Furniture', 49.99, 75),
('USB Cable', 'Accessories', 9.99, 200);

-- Create orders table
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    product_id INT,
    quantity INT,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Insert sample order data
INSERT INTO orders (user_id, product_id, quantity, total_amount) VALUES
(1, 1, 1, 1299.99),
(2, 2, 2, 59.98),
(1, 3, 1, 249.99),
(3, 4, 3, 149.97),
(4, 5, 5, 49.95),
(5, 1, 1, 1299.99),
(2, 3, 2, 499.98);

-- End of monthly backup