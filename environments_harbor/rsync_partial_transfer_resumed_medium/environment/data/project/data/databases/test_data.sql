DROP DATABASE IF EXISTS test_db;
CREATE DATABASE test_db;
USE test_db;

-- Table structure for users
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    date_registered DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    is_active BOOLEAN DEFAULT TRUE,
    user_role ENUM('admin', 'manager', 'customer') DEFAULT 'customer'
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(user_role);

-- Table structure for customers
CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    company_name VARCHAR(100),
    phone VARCHAR(20),
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(50),
    postal_code VARCHAR(20),
    country VARCHAR(50) DEFAULT 'USA',
    credit_limit DECIMAL(10,2) DEFAULT 5000.00,
    account_balance DECIMAL(10,2) DEFAULT 0.00,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_customers_user ON customers(user_id);
CREATE INDEX idx_customers_city ON customers(city);

-- Table structure for products
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_code VARCHAR(50) UNIQUE NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    description TEXT,
    category VARCHAR(50),
    unit_price DECIMAL(10,2) NOT NULL,
    cost_price DECIMAL(10,2),
    stock_quantity INT DEFAULT 0,
    reorder_level INT DEFAULT 10,
    supplier_id INT,
    is_available BOOLEAN DEFAULT TRUE,
    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_code ON products(product_code);
CREATE INDEX idx_products_available ON products(is_available);

-- Table structure for orders
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    order_number VARCHAR(50) UNIQUE NOT NULL,
    customer_id INT NOT NULL,
    order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    shipped_date DATETIME,
    delivery_date DATETIME,
    order_status ENUM('pending', 'processing', 'shipped', 'delivered', 'cancelled') DEFAULT 'pending',
    payment_method VARCHAR(50),
    subtotal DECIMAL(10,2),
    tax_amount DECIMAL(10,2),
    shipping_cost DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    notes TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_status ON orders(order_status);
CREATE INDEX idx_orders_date ON orders(order_date);

-- Table structure for order_items
CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY AUTO_INCREMENT,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    discount_percent DECIMAL(5,2) DEFAULT 0.00,
    line_total DECIMAL(10,2),
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);

-- Table structure for transactions
CREATE TABLE transactions (
    transaction_id INT PRIMARY KEY AUTO_INCREMENT,
    transaction_ref VARCHAR(100) UNIQUE NOT NULL,
    order_id INT,
    customer_id INT NOT NULL,
    transaction_type ENUM('payment', 'refund', 'credit', 'debit') DEFAULT 'payment',
    amount DECIMAL(10,2) NOT NULL,
    transaction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    payment_gateway VARCHAR(50),
    status ENUM('pending', 'completed', 'failed', 'reversed') DEFAULT 'pending',
    description TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE INDEX idx_transactions_customer ON transactions(customer_id);
CREATE INDEX idx_transactions_order ON transactions(order_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_transactions_status ON transactions(status);

-- Table structure for suppliers
CREATE TABLE suppliers (
    supplier_id INT PRIMARY KEY AUTO_INCREMENT,
    supplier_name VARCHAR(100) NOT NULL,
    contact_person VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    address VARCHAR(255),
    city VARCHAR(100),
    country VARCHAR(50),
    payment_terms VARCHAR(100),
    rating DECIMAL(3,2) DEFAULT 5.00
);

CREATE INDEX idx_suppliers_name ON suppliers(supplier_name);

-- Insert data into users
INSERT INTO users (username, email, password_hash, first_name, last_name, last_login, user_role) VALUES
('jsmith', 'john.smith@email.com', 'a1b2c3d4e5f6g7h8i9j0', 'John', 'Smith', '2024-01-15 10:30:00', 'customer'),
('mjohnson', 'mary.johnson@email.com', 'b2c3d4e5f6g7h8i9j0k1', 'Mary', 'Johnson', '2024-01-14 14:20:00', 'customer'),
('bwilliams', 'bob.williams@email.com', 'c3d4e5f6g7h8i9j0k1l2', 'Bob', 'Williams', '2024-01-16 09:15:00', 'manager'),
('sjones', 'sarah.jones@email.com', 'd4e5f6g7h8i9j0k1l2m3', 'Sarah', 'Jones', '2024-01-13 16:45:00', 'customer'),
('mbrown', 'michael.brown@email.com', 'e5f6g7h8i9j0k1l2m3n4', 'Michael', 'Brown', '2024-01-17 11:00:00', 'customer'),
('ldavis', 'lisa.davis@email.com', 'f6g7h8i9j0k1l2m3n4o5', 'Lisa', 'Davis', '2024-01-12 13:30:00', 'customer'),
('dmiller', 'david.miller@email.com', 'g7h8i9j0k1l2m3n4o5p6', 'David', 'Miller', '2024-01-18 08:20:00', 'admin'),
('jwilson', 'jennifer.wilson@email.com', 'h8i9j0k1l2m3n4o5p6q7', 'Jennifer', 'Wilson', '2024-01-11 15:10:00', 'customer'),
('rmoore', 'robert.moore@email.com', 'i9j0k1l2m3n4o5p6q7r8', 'Robert', 'Moore', '2024-01-19 12:40:00', 'customer'),
('ktaylor', 'karen.taylor@email.com', 'j0k1l2m3n4o5p6q7r8s9', 'Karen', 'Taylor', '2024-01-10 10:55:00', 'customer'),
('tanderson', 'thomas.anderson@email.com', 'k1l2m3n4o5p6q7r8s9t0', 'Thomas', 'Anderson', '2024-01-20 14:15:00', 'customer'),
('nthomas', 'nancy.thomas@email.com', 'l2m3n4o5p6q7r8s9t0u1', 'Nancy', 'Thomas', '2024-01-09 09:30:00', 'customer'),
('cjackson', 'charles.jackson@email.com', 'm3n4o5p6q7r8s9t0u1v2', 'Charles', 'Jackson', '2024-01-21 11:25:00', 'manager'),
('bwhite', 'barbara.white@email.com', 'n4o5p6q7r8s9t0u1v2w3', 'Barbara', 'White', '2024-01-08 16:00:00', 'customer'),
('jharris', 'james.harris@email.com', 'o5p6q7r8s9t0u1v2w3x4', 'James', 'Harris', '2024-01-22 13:45:00', 'customer'),
('smartin', 'susan.martin@email.com', 'p6q7r8s9t0u1v2w3x4y5', 'Susan', 'Martin', '2024-01-07 10:10:00', 'customer'),
('jthompson', 'joseph.thompson@email.com', 'q7r8s9t0u1v2w3x4y5z6', 'Joseph', 'Thompson', '2024-01-23 15:30:00', 'customer'),
('lgarcia', 'linda.garcia@email.com', 'r8s9t0u1v2w3x4y5z6a7', 'Linda', 'Garcia', '2024-01-06 08:50:00', 'customer'),
('cmartinez', 'christopher.martinez@email.com', 's9t0u1v2w3x4y5z6a7b8', 'Christopher', 'Martinez', '2024-01-24 12:20:00', 'customer'),
('mrobinson', 'michelle.robinson@email.com', 't0u1v2w3x4y5z6a7b8c9', 'Michelle', 'Robinson', '2024-01-05 14:40:00', 'customer'),
('dclark', 'daniel.clark@email.com', 'u1v2w3x4y5z6a7b8c9d0', 'Daniel', 'Clark', '2024-01-25 09:05:00', 'customer'),
('drodriguez', 'donna.rodriguez@email.com', 'v2w3x4y5z6a7b8c9d0e1', 'Donna', 'Rodriguez', '2024-01-04 11:35:00', 'customer'),
('plewis', 'paul.lewis@email.com', 'w3x4y5z6a7b8c9d0e1f2', 'Paul', 'Lewis', '2024-01-26 16:15:00', 'customer'),
('clee', 'carol.lee@email.com', 'x4y5z6a7b8c9d0e1f2g3', 'Carol', 'Lee', '2024-01-03 13:00:00', 'customer'),
('mwalker', 'mark.walker@email.com', 'y5z6a7b8c9d0e1f2g3h4', 'Mark', 'Walker', '2024-01-27 10:45:00', 'customer'),
('rhall', 'ruth.hall@email.com', 'z6a7b8c9d0e1f2g3h4i5', 'Ruth', 'Hall', '2024-01-02 15:20:00', 'customer'),
('sallen', 'steven.allen@email.com', 'a7b8c9d0e1f2g3h4i5j6', 'Steven', 'Allen', '2024-01-28 08:30:00', 'customer'),
('byoung', 'betty.young@email.com', 'b8c9d0e1f2g3h4i5j6k7', 'Betty', 'Young', '2024-01-01 12:10:00', 'customer'),
('eking', 'edward.king@email.com', 'c9d0e1f2g3h4i5j6k7l8', 'Edward', 'King', '2024-01-29 14:50:00', 'customer'),
('dwright', 'dorothy.wright@email.com', 'd0e1f2g3h4i5j6k7l8m9', 'Dorothy', 'Wright', '2023-12-31 09:25:00', 'customer'),
('glopez', 'george.lopez@email.com', 'e1f2g3h4i5j6k7l8m9n0', 'George', 'Lopez', '2024-01-30 11:40:00', 'customer'),
('shill', 'sandra.hill@email.com', 'f2g3h4i5j6k7l8m9n0o1', 'Sandra', 'Hill', '2023-12-30 16:05:00', 'customer'),
('kscott', 'kenneth.scott@email.com', 'g3h4i5j6k7l8m9n0o1p2', 'Kenneth', 'Scott', '2024-01-31 13:15:00', 'customer'),
('hgreen', 'helen.green@email.com', 'h4i5j6k7l8m9n0o1p2q3', 'Helen', 'Green', '2023-12-29 10:30:00', 'customer'),
('badams', 'brian.adams@email.com', 'i5j6k7l8m9n0o1p2q3r4', 'Brian', 'Adams', '2024-02-01 15:45:00', 'customer'),
('sbaker', 'sharon.baker@email.com', 'j6k7l8m9n0o1p2q3r4s5', 'Sharon', 'Baker', '2023-12-28 08:55:00', 'customer'),
('rgonzalez', 'ronald.gonzalez@email.com', 'k7l8m9n0o1p2q3r4s5t6', 'Ronald', 'Gonzalez', '2024-02-02 12:25:00', 'customer'),
('knelson', 'kimberly.nelson@email.com', 'l8m9n0o1p2q3r4s5t6u7', 'Kimberly', 'Nelson', '2023-12-27 14:35:00', 'customer'),
('acarter', 'anthony.carter@email.com', 'm9n0o1p2q3r4s5t6u7v8', 'Anthony', 'Carter', '2024-02-03 09:50:00', 'customer'),
('mmitchell', 'margaret.mitchell@email.com', 'n0o1p2q3r4s5t6u7v8w9', 'Margaret', 'Mitchell', '2023-12-26 11:15:00', 'customer'),
('jperez', 'jason.perez@email.com', 'o1p2q3r4s5t6u7v8w9x0', 'Jason', 'Perez', '2024-02-04 16:20:00', 'customer'),
('eroberts', 'emily.roberts@email.com', 'p2q3r4s5t6u7v8w9x0y1', 'Emily', 'Roberts', '2023-12-25 13:40:00', 'customer'),
('mturner', 'matthew.turner@email.com', 'q3r4s5t6u7v8w9x0y1z2', 'Matthew', 'Turner', '2024-02-05 10:05:00', 'customer'),
('dphillips', 'deborah.phillips@email.com', 'r4s5t6u7v8w9x0y1z2a3', 'Deborah', 'Phillips', '2023-12-24 15:30:00', 'customer'),
('rcampbell', 'ryan.campbell@email.com', 's5t6u7v8w9x0y1z2a3b4', 'Ryan', 'Campbell', '2024-02-06 08:45:00', 'customer'),
('lstephanie', 'laura.stephanie@email.com', 't6u7v8w9x0y1z2a3b4c5', 'Laura', 'Stephanie', '2023-12-23 12:55:00', 'customer'),
('jparker', 'jeffrey.parker@email.com', 'u7v8w9x0y1z2a3b4c5d6', 'Jeffrey', 'Parker', '2024-02-07 14:10:00', 'customer'),
('revans', 'rebecca.evans@email.com', 'v8w9x0y1z2a3b4c5d6e7', 'Rebecca', 'Evans', '2023-12-22 09:20:00', 'customer'),
('gedwards', 'gary.edwards@email.com', 'w9x0y1z2a3b4c5d6e7f8', 'Gary', 'Edwards', '2024-02-08 11:35:00', 'manager'),
('scollins', 'stephanie.collins@email.com', 'x0y1z2a3b4c5d6e7f8g9', 'Stephanie', 'Collins', '2023-12-21 16:50:00', 'customer'),
('nstewart', 'nicholas.stewart@email.com', 'y1z2a3b4c5d6e7f8g9h0', 'Nicholas', 'Stewart', '2024-02-09 13:05:00', 'customer'),
('vmorris', 'virginia.morris@email.com', 'z2a3b4c5d6e7f8g9h0i1', 'Virginia', 'Morris', '2023-12-20 10:15:00', 'customer'),
('jrogers', 'jonathan.rogers@email.com', 'a3b4c5d6e7f8g9h0i1j2', 'Jonathan', 'Rogers', '2024-02-10 15:25:00', 'customer'),
('pradison', 'pamela.radison@email.com', 'b4c5d6e7f8g9h0i1j2k3', 'Pamela', 'Radison', '2023-12-19 08:40:00', 'customer'),
('hjones', 'harold.jones@email.com', 'c5d6e7f8g9h0i1j2k3l4', 'Harold', 'Jones', '2024-02-11 12:00:00', 'customer');

-- Insert data into customers
INSERT INTO customers (user_id, company_name, phone, address_line1, city, state, postal_code, country, credit_limit, account_balance) VALUES
(1, 'Smith Enterprises', '555-0101', '123 Main St', 'New York', 'NY', '10001', 'USA', 10000.00, 2500.50),
(2, 'Johnson Corp', '555-0102', '456 Oak Ave', 'Los Angeles', 'CA', '90001', 'USA', 15000.00, 3200.75),
(4, 'Jones & Associates', '555-0104', '789 Pine Rd', 'Chicago', 'IL', '60601', 'USA', 8000.00, 1800.00),
(5, 'Brown Industries', '555-0105', '321 Elm St', 'Houston', 'TX', '77001', 'USA', 12000.00, 4100.25),
(6, 'Davis Solutions', '555-0106', '654 Maple Dr', 'Phoenix', 'AZ', '85001', 'USA', 9000.00, 2200.00),
(8, 'Wilson Trading', '555-0108', '987 Cedar Ln', 'Philadelphia', 'PA', '19019', 'USA', 11000.00, 3500.50),
(9, 'Moore Systems', '555-0109', '147 Birch Ct', 'San Antonio', 'TX', '78201', 'USA', 7500.00, 1500.00),
(10, 'Taylor Logistics', '555-0110', '258 Spruce Way', 'San Diego', 'CA', '92101', 'USA', 13000.00, 4800.75),
(11, 'Anderson Tech', '555-0111', '369 Walnut Blvd', 'Dallas', 'TX', '75201', 'USA', 16000.00, 5200.00),
(12, 'Thomas Manufacturing', '555-0112', '741 Hickory Ave', 'San Jose', 'CA', '95101', 'USA', 14000.00, 3900.25),
(14, 'White Consulting', '555-0114', '852 Ash St', 'Jacksonville', 'FL', '32099', 'USA', 9500.00, 2100.50),
(15, 'Harris Distribution', '555-0115', '963 Poplar Rd', 'Fort Worth', 'TX', '76101', 'USA', 10500.00, 2800.00),
(16, 'Martin Services', '555-0116', '159 Willow Dr', 'Columbus', 'OH', '43004', 'USA', 8500.00, 1900.75),
(17, 'Thompson Retail', '555-0117', '357 Cherry Ln', 'Charlotte', 'NC', '28201', 'USA', 12500.00, 3600.00),
(18, 'Garcia Imports', '555-0118', '486 Beech Ct', 'Indianapolis', 'IN', '46201', 'USA', 11500.00, 3100.25),
(19, 'Martinez Export', '555-0119', '624 Fir Way', 'San Francisco', 'CA', '94101', 'USA', 15500.00, 4500.50),
(20, 'Robinson Holdings', '555-0120', '735 Hemlock Blvd', 'Seattle', 'WA', '98101', 'USA', 17000.00, 5800.00),
(21, 'Clark Ventures', '555-0121', '846 Redwood Ave', 'Denver', 'CO', '80201', 'USA', 13500.00, 4200.75),
(22, 'Rodriguez Partners', '555-0122', '957 Cypress St', 'Washington', 'DC', '20001', 'USA', 16500.00, 5100.00),
(23, 'Lewis Group', '555-0123', '168 Magnolia Rd', 'Boston', 'MA', '02101', 'USA', 14500.00, 3800.25),
(24, 'Lee Enterprises', '555-0124', '279 Dogwood Dr', 'Nashville', 'TN', '37201', 'USA', 9800.00, 2300.50),
(25, 'Walker Industries', '555-0125', '381 Sycamore Ln', 'Detroit', 'MI', '48201', 'USA', 10800.00, 2900.00),
(26, 'Hall Corporation', '555-0126', '492 Cottonwood Ct', 'El Paso', 'TX', '79901', 'USA', 8800.00, 1700.75),
(27, 'Allen Systems', '555-0127', '513 Juniper Way', 'Memphis', 'TN', '37501', 'USA', 12800.00, 3700.00),
(28, 'Young Trading', '555-0128', '624 Mesquite Blvd', 'Portland', 'OR', '97201', 'USA', 11800.00, 3200.25),
(29, 'King Solutions', '555-0129', '735 Oakwood Ave', 'Oklahoma City', 'OK', '73101', 'USA', 9200.00, 2000.50),
(30, 'Wright Logistics', '555-0130', '846 Cedarwood St', 'Las Vegas', 'NV', '89101', 'USA', 13200.00, 4300.00),
(31, 'Lopez Tech', '555-0131', '957 Pinewood Rd', 'Louisville', 'KY', '40201', 'USA', 15800.00, 4900.75),
(32, 'Hill Manufacturing', '555-0132', '168 Elmwood Dr', 'Baltimore', 'MD', '21201', 'USA', 14200.00, 3900.00),
(33, 'Scott Consulting', '555-0133', '279 Maplewood Ln', 'Milwaukee', 'WI', '53201', 'USA', 10200.00, 2600.25),
(34, 'Green Distribution', '555-0134', '381 Ashwood Ct', 'Albuquerque', 'NM', '87101', 'USA', 9600.00, 2200.50),
(35, 'Adams Services', '555-0135', '492 Birchwood Way', 'Tucson', 'AZ', '85701', 'USA', 11200.00, 3000.00),
(36, 'Baker Retail', '555-0136', '513 Hickorywood Blvd', 'Fresno', 'CA', '93650', 'USA', 8200.00, 1600.75),
(37, 'Gonzalez Imports', '555-0137', '624 Walnutwood Ave', 'Sacramento', 'CA', '94201', 'USA', 12200.00, 3500.00),
(38, 'Nelson Export', '555-0138', '735 Sprucewood St', 'Long Beach', 'CA', '90801', 'USA', 13800.00, 4100.25),
(39, 'Carter Holdings', '555-0139', '846 Firwood Rd', 'Kansas City', 'MO', '64101', 'USA', 15200.00, 4700.50),
(40, 'Mitchell Ventures', '555-0140', '957 Hemlockwood Dr', 'Mesa', 'AZ', '85201', 'USA', 16800.00, 5300.00),
(41, 'Perez Partners', '555-0141', '168 Redwoodwood Ln', 'Virginia Beach', 'VA', '23450', 'USA', 14800.00, 4000.75),
(42, 'Roberts Group', '555-0142', '279 Cypresswood Ct', 'Atlanta', 'GA', '30301', 'USA', 17200.00, 5600.00),
(43, 'Turner Enterprises', '555-0143', '381 Magnoliawood Way', 'Colorado Springs', 'CO', '80901', 'USA', 13800.00, 4200.25),
(44, 'Phillips Industries', '555-0144', '492 Dogwoodwood Blvd', 'Raleigh', 'NC', '27601', 'USA', 10600.00, 2700.50),
(45, 'Campbell Corporation', '555-0145', '513 Sycamorewood Ave', 'Omaha', 'NE', '68101', 'USA', 9400.00, 2100.00),
(46, 'Stephanie Systems', '555-0146', '624 Cottonwoodwood St', 'Miami', 'FL', '33101', 'USA', 11600.00, 3100.75),
(47, 'Parker Trading', '555-0147', '735 Juniperhood Rd', 'Oakland', 'CA', '94601', 'USA', 12600.00, 3600.00),
(48, 'Evans Solutions', '555-0148', '846 Mesquitewood Dr', 'Tulsa', 'OK', '74101', 'USA', 10400.00, 2800.25),
(50, 'Collins Logistics', '555-0150', '168 Cedarwoodwood Way', 'Minneapolis', 'MN', '55401', 'USA', 14600.00, 4400.50),
(51, 'Stewart Tech', '555-0151', '279 Pinewoodwood Blvd', 'Wichita', 'KS', '67201', 'USA', 16200.00, 5000.00),
(52, 'Morris Manufacturing', '555-0152', '381 Elmwoodwood Ave', 'Arlington', 'TX', '76001', 'USA', 15600.00, 4800.75);

-- Insert data into suppliers
INSERT INTO suppliers (supplier_name, contact_person, email, phone, address, city, country, payment_terms, rating) VALUES
('Global Tech Supply', 'John Anderson', 'janderson@globaltech.com', '555-1001', '100 Tech Park', 'San Jose', 'USA', 'Net 30', 4.5),
('MegaParts Inc', 'Sarah Wilson', 'swilson@megaparts.com', '555-1002', '200 Industrial Blvd', 'Detroit', 'USA', 'Net 45', 4.8),
('Premier Components', 'Mike Davis', 'mdavis@premiercomp.com', '555-1003', '300 Manufacturing Way', 'Chicago', 'USA', 'Net 60', 4.2),
('Quality Wholesale', 'Lisa Brown', 'lbrown@qualitywholesale.com', '555-1004', '400 Distribution Center', 'Dallas', 'USA', 'Net 30', 4.6),
('Supreme Suppliers', 'Robert Johnson', 'rjohnson@supremesupply.com', '555-1005', '500 Warehouse Rd', 'Los Angeles', 'USA', 'Net 45', 4.9);

-- Insert data into products
INSERT INTO products (product_code, product_name, description, category, unit_price, cost_price, stock_quantity, reorder_level, supplier_id, is_available) VALUES
('PROD-001', 'Wireless Mouse', 'Ergonomic wireless mouse with USB receiver', 'Electronics', 29.99, 15.00, 150, 20, 1, TRUE),
('PROD-002', 'Mechanical Keyboard', 'RGB backlit mechanical gaming keyboard', 'Electronics', 89.99, 45.00, 75, 15, 1, TRUE),
('PROD-003', 'USB-C Cable', '6ft USB-C to USB-A charging cable', 'Accessories', 12.99, 6.50, 300, 50, 2, TRUE),
('PROD-004', 'Laptop Stand', 'Adjustable aluminum laptop stand', 'Accessories', 39.99, 20.00, 100, 15, 3, TRUE),
('PROD-005', 'Wireless Headphones', 'Noise-cancelling Bluetooth hea