-- Database Backup
-- Generated: 2024-01-15
-- Database: production_db

CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL(10,2),
    category_id INT
);

INSERT INTO products VALUES (1, 'Widget A', 19.99, 1);
INSERT INTO products VALUES (2, 'Widget B', 29.99, 1);
INSERT INTO products VALUES (3, 'Gadget Pro', 49.99, 2);
INSERT INTO products VALUES (4, 'Tool Set', 89.99, 3);
INSERT INTO products VALUES (5, 'Premium Kit', 129.99, 2);

CREATE TABLE categories (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    description TEXT
);

INSERT INTO categories VALUES (1, 'Electronics', 'Electronic devices and accessories');
INSERT INTO categories VALUES (2, 'Tools', 'Hardware and tool equipment');
INSERT INTO categories VALUES (3, 'Home & Garden', 'Products for home improvement');

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10,2)
);

INSERT INTO orders VALUES (1001, 501, '2024-01-10', 149.98);
INSERT INTO orders VALUES (1002, 502, '2024-01-11', 89.99);
INSERT INTO orders VALUES (1003, 503, '2024-01-12', 49.99);