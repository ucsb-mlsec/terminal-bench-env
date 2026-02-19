-- Weekly Database Backup - Week 02
-- Generated: 2024-01-15
-- Database: production_db
-- Backup Type: Full Database Dump
-- 
-- This is an automated weekly backup containing all production tables and data
-- ============================================================================

-- Disable foreign key checks for faster import
SET FOREIGN_KEY_CHECKS=0;

-- Table structure for users
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `email` varchar(100) NOT NULL,
  `created_at` timestamp DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Insert sample user data
INSERT INTO `users` (`id`, `username`, `email`) VALUES
(1, 'admin', 'admin@example.com'),
(2, 'jdoe', 'jdoe@example.com'),
(3, 'msmith', 'msmith@example.com');

-- Table structure for orders
DROP TABLE IF EXISTS `orders`;
CREATE TABLE `orders` (
  `order_id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `total_amount` decimal(10,2) NOT NULL,
  `order_date` datetime DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`order_id`),
  KEY `user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Insert sample order data
INSERT INTO `orders` (`order_id`, `user_id`, `total_amount`, `order_date`) VALUES
(101, 1, 299.99, '2024-01-10 10:30:00'),
(102, 2, 149.50, '2024-01-12 14:22:00'),
(103, 3, 89.99, '2024-01-14 09:15:00');

-- Re-enable foreign key checks
SET FOREIGN_KEY_CHECKS=1;

-- End of weekly backup for week 02