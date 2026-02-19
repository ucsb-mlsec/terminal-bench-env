-- Database Backup File
-- Generated: 2024-01-19 03:00:00
-- Database: production_db
-- Type: Daily Backup
-- Server: db-prod-01

-- Backup Start
SET FOREIGN_KEY_CHECKS=0;
SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";

-- Drop existing tables if they exist
DROP TABLE IF EXISTS `users`;
DROP TABLE IF EXISTS `orders`;
DROP TABLE IF EXISTS `products`;

-- Create users table
CREATE TABLE `users` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `email` varchar(100) NOT NULL,
  `created_at` timestamp DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Insert user data
INSERT INTO `users` (`id`, `username`, `email`) VALUES
(1, 'admin', 'admin@example.com'),
(2, 'john_doe', 'john@example.com'),
(3, 'jane_smith', 'jane@example.com');

-- Create products table
CREATE TABLE `products` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `price` decimal(10,2) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Insert product data
INSERT INTO `products` (`id`, `name`, `price`) VALUES
(1, 'Widget A', 29.99),
(2, 'Widget B', 49.99);

-- Create orders table
CREATE TABLE `orders` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `product_id` int(11) NOT NULL,
  `quantity` int(11) DEFAULT 1,
  `order_date` timestamp DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Insert order data
INSERT INTO `orders` (`id`, `user_id`, `product_id`, `quantity`) VALUES
(1, 2, 1, 3),
(2, 3, 2, 1);

SET FOREIGN_KEY_CHECKS=1;

-- Backup completed successfully
-- End of backup file