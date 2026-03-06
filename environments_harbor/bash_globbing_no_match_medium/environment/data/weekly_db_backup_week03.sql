-- Weekly Database Backup - Week 03
-- Generated: 2024-01-15
-- Database: production_db
-- Backup Type: Weekly Full Backup
-- ================================================

-- Table structure for users
DROP TABLE IF EXISTS users;
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert data for users
INSERT INTO users (user_id, username, email) VALUES
(1, 'admin', 'admin@example.com'),
(2, 'john_doe', 'john@example.com'),
(3, 'jane_smith', 'jane@example.com'),
(4, 'bob_wilson', 'bob@example.com');

-- Table structure for orders
DROP TABLE IF EXISTS orders;
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10,2),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Insert data for orders
INSERT INTO orders (order_id, user_id, order_date, total_amount) VALUES
(101, 2, '2024-01-10 10:30:00', 299.99),
(102, 3, '2024-01-11 14:15:00', 149.50),
(103, 2, '2024-01-13 09:45:00', 89.99),
(104, 4, '2024-01-14 16:20:00', 450.00);

-- Table structure for products
DROP TABLE IF EXISTS products;
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    stock_quantity INT DEFAULT 0
);

-- Insert data for products
INSERT INTO products (product_id, product_name, price, stock_quantity) VALUES
(1001, 'Laptop Pro 15', 1299.99, 25),
(1002, 'Wireless Mouse', 29.99, 150),
(1003, 'USB-C Cable', 19.99, 200),
(1004, 'Monitor 27 inch', 399.99, 40);

-- End of backup