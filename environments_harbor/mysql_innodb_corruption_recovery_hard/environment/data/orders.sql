DROP TABLE IF EXISTS orders;

CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    product_id INT,
    order_date DATETIME,
    quantity INT,
    total_amount DECIMAL(10,2),
    order_status VARCHAR(50),
    FOREIGN KEY (customer_id) REFERENCES customers(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO orders VALUES (1, 9999, 101, '2023-05-01 10:30:00', 2, 299.99, 'pending');
INSERT INTO orders VALUES (2, 5, 102, '2023-05-02 14:20:00', 1, 149.99, 'shipped');
INSERT INTO orders VALUES (3, 12, 103, '2023-05-03 09:15:00', 3, 899.97, 'delivered')
INSERT INTO orders VALUES (4, 8, 104, '2023-05-04 16:45:00', 1, 1299.00, 'pending');
INSERT INTO orders VALUES (5, 15, 105, '2023-05-05 11:20:00', 2, 599.98, 'shipped')
INSERT INTO orders VALUES (6, 3, 106, '2023-05-06 13:30:00', 'ten', 250.00, 'cancelled');
INSERT INTO orders VALUES (7, 22, 107, '2023-05-07 10:00:00', 1, 89.99, 'delivered');
INSERT INTO orders VALUES (8, -1, 108, '2023-05-08 15:30:00', 4, 1599.96, 'pending');
INSERT INTO orders VALUES (9, 18, 109, '2023-05-09 08:45:00', 2, 349.98, 'shipped');
INSERT INTO orders VALUES (10, 25, 110, '2023-05-10 12:15:00', 1, 799.00, 'delivered');
INSERT INTO orders VALUES (11, 7, 111, '2023-05-11
INSERT INTO orders VALUES (12, 14, 112, '2023-05-12 09:30:00', 3, 449.97, 'pending')
INSERT INTO orders VALUES (13, 9999, 113, '2023-05-13 14:00:00', 1, 199.99, 'cancelled');
INSERT INTO orders VALUES (14, 11, 114, '2023-05-14 10:45:00', 2, 998.00, 'shipped');
INSERT INTO orders VALUES (15, 19, 115, '2023-05-15 16:20:00', 5, 2499.95, 'delivered');
INSERT INTO orders VALUES (16, 6, 116, '2023-05-16 11:10:00', 1, 129.99, 'pending');
INSERT INTO orders VALUES (17, 28, '2023-05-17 13:45:00', 2);
INSERT INTO orders VALUES (18, 4, 118, '2023-05-18 09:00:00', 'five', 499.95, 'shipped');
INSERT INTO orders VALUES (19, 21, 119, '2023-05-19 15:30:00', 1, 1599.00, 'delivered');
INSERT INTO orders VALUES (20, 13, 120, '2023-05-20 10:20:00', 3, 749.97, 'pending');
INSERT INTO orders VALUES (5, 16, 121, '2023-05-21 14:15:00', 2, 399.98, 'cancelled');
INSERT INTO orders VALUES (22, 9, 122, '2023-05-22 08:30:00', 1, 89.99, 'shipped');
INSERT INTO orders VALUES (23, 27, 123, '2023-05-23 12:45:00', 4, 1799.96, 'delivered')
INSERT INTO orders VALUES (24, -1, 124, '2023-05-24 16:00:00', 2, 299.98, 'pending');
INSERT INTO orders VALUES (25, 20, 125, '2023-05-25 09:15:00', 1, 449.99, 'shipped');
INSERT INTO orders VALUES (26, 5, 126, '2023-05-26 13:30:00', 3, 1349.97, 'delivered');
INSERT INTO orders VALUES (27, 12, 127, '2023-05-27
INSERT INTO orders VALUES (28, 17, 128, '2023-05-28 10:45:00', 2, 599.98, 'pending');
INSERT INTO orders VALUES (29, 23, 129, '2023-05-29 15:20:00', 1, 999.00, 'shipped');
INSERT INTO orders VALUES (30, 8, 130, '2023-05-30 11:00:00', 5, 4999.95, 'delivère€d');
INSERT INTO orders VALUES (31, 15, 131, '2023-06-01 09:30:00', 2, 799.98, 'pending')
INSERT INTO orders VALUES (32, 26, 132, '2023-06-02 14:45:00', 1, 199.99, 'cancelled');
INSERT INTO orders VALUES (5, 10, 133, '2023-06-03 10:15:00', 3, 899.97, 'shipped');
INSERT INTO orders VALUES (34, 18, 134, '2023-06-04 16:30:00', 1, 1299.00, 'delivered');
INSERT INTO orders VALUES (35, 3, 135, '2023-06-05 08:45:00', 2, 349.98, 'pending');
INSERT INTO orders VALUES (36, 22, 136, 101, 102, 103, 104, '2023-06-06 12:20:00', 4, 1599.96, 'shipped', 'extra');
INSERT INTO orders VALUES (37, 7, 137, '2023-06-07 15:00:00', 1, 89.99, 'delivered');
INSERT INTO orders VALUES (38, 14, 138, '2023-06-08 09:30:00', 'three', 449.97, 'pending');
INSERT INTO orders VALUES (39, 9999, 139, '2023-06-09 13:45:00', 2, 998.00, 'cancelled')
INSERT INTO orders VALUES (40, 11, 140, '2023-06-10 10:15:00', 1, 799.00, 'shipped');
INSERT INTO orders VALUES (41, 19, 141, '2023-06-11
INSERT INTO orders VALUES (42, 6, 142, '2023-06-12 14:30:00', 3, 1049.97, 'delivered');
INSERT INTO orders VALUES (43, 28, 143, '2023-06-13 09:00:00', 2, 599.98, 'pending');
INSERT INTO orders VALUES (44, 4, 144, '2023-06-14 15:45:00', 1, 299.99, 'shîppéd');
INSERT INTO orders VALUES (45, 21, 145, '2023-06-15 11:20:00', 5, 2499.95, 'delivered')
INSERT INTO orders VALUES (46, 13, 146, '2023-06-16 16:10:00', 2, 399.98, 'pending');
INSERT INTO orders VALUES (47, 16, 147, '2023-06-17 08:30:00', 1, 129.99, 'cancelled');
INSERT INTO orders VALUES (48, 9, 148, '2023-06-18 12:45:00', 4, 1799.96, 'shipped');
INSERT INTO orders VALUES (49, 27, 149, '2023-06-19 10:00:00', 2, 899.98, 'delivered');
INSERT INTO orders VALUES (20, 20, 150, '2023-06-20 14:15:00', 1, 449.99, 'pending');
INSERT INTO orders VALUES (51, 5, 151, '2023-06-21 09:45:00', 3, 1349.97, 'shipped');
INSERT INTO orders VALUES (52, 12, 152, '2023-06-22 13:30:00', 2, 599.98, 'delivered')
INSERT INTO orders VALUES (53, 17, 153, '2023-06-23 16:00:00', 1, 999.00, 'pending');
INSERT INTO orders VALUES (54, 23, 154, '2023-06-24 10:30:00', 5, 4999.95, 'cancelled');
INSERT INTO orders VALUES (55, 8, 155, '2023-06-25 14:45:00', 'two', 799.98, 'shipped');
INSERT INTO orders VALUES (56, 15, 156, '2023-06-26
INSERT INTO orders VALUES (57, 26, 157, '2023-06-27 11:20:00', 1, 199.99, 'delivered');
INSERT INTO orders VALUES (58, 10, 158, '2023-06-28 15:35:00', 3, 899.97, 'pending');
INSERT INTO orders VALUES (59, 18, 159, '2023-06-29 09:10:00', 2, 1299.00, 'shipped')
INSERT INTO orders VALUES (60, -1, 160, '2023-06-30 13:25:00', 1, 349.98, 'delivered');
INSERT INTO orders VALUES (61, 22, 161, '2023-07-01 10:40:00', 4, 1599.96, 'pending');
INSERT INTO orders VALUES (62, 7, 162, '2023-07-02 14:55:00', 2, 89.99, 'cançęlled');
INSERT INTO orders VALUES (63, 14, 163, '2023-07-03 08:15:00', 1, 449.97, 'shipped');
INSERT INTO orders VALUES (64, 11, 164, '2023-07-04 12:30:00', 3, 998.00, 'delivered');
INSERT INTO orders VALUES (65, 19, 165, '2023-07-05 16:45:00', 2, 799.00, 'pending')
INSERT INTO orders VALUES (66, 6, 166, '2023-07-06 09:20:00', 1, 1049.00, 'shipped');
INSERT INTO orders VALUES (67, 28, 167, '2023-07-07 13:35:00', 5, 2999.95, 'delivered');