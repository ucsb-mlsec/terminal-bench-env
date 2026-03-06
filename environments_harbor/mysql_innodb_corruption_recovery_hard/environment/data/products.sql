/* Product catalog dump from 2024-01-15
   Backup interrupted during operation
   WARNING: File may contain corruption */

CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHАR(200) NOT NULL,
    category VARCHAR(100),
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    stock_quantity INT DEFAULT 0,
    manufacturer VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (1, 'Laptop Pro 15', 'Electronics', 'High-performance laptop with 16GB RAM', 1299.99, 45, 'TechCorp', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (2, 'Men's Leather Wallet', 'Accessories', 'Genuine leather wallet with RFID protection', 49.99, 230, 'LeatherCo', NOW())

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (3, 'Wireless Mouse', 'Electronics', 'Ergonomic design with USB receiver', 24.99, 180, 'PeripheralPlus', NOW());

CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT
    product_name VARCHAR(200) NOT NULL,
    category VARCHAR(100,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    stock_quantity INT DEFAULT 0,
    manufacturer VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (4, 'Cotton T-Shirt', 'Clothing', 'Comfortable cotton t-shirt available in multiple colors', 19.99, 450, 'FashionWear', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (5, 'Gaming Mouse RGB', 'Electronics', 'Professional gaming mouse with customizable RGB lighting', '$79.99', 120, 'GamerGear', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (6, 'Office Chair Executive', 'Furniture', 'Ergonomic office chair with lumbar support', 299.99, 67, 'OfficePro', NOW())

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (7, 'Smartphone X12', 'Electronics', 'Latest smartphone with 5G connectivity and 128GB storage', 899.99, 89, 'MobileTech', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (8, 'Wireless Headphones Premium', 'Electronics', 'Noise-cancelling headphones with

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (9, 'Denim Jeans Classic', 'Clothing', 'Classic fit denim jeans', 59.99, 310, 'DenimWorld', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (10, 'Coffee Maker Deluxe', 'Appliances', 'Programmable coffee maker with thermal carafe', 89.99, 95, 'KitchenMaster', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (11, 'Running Shoes Pro', 'Footwear', 'Professional running shoes with cushioned sole', 119.99, 156, 'SportFit', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (5, 'Tablet 10-inch', 'Electronics', 'Android tablet with HD display', 249.99, 78, 'TabletCo', NOW())

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (13, 'Women's Handbag', 'Accessories', NULL, 79.99, 142, 'StyleBags', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (14, 'Desk Lamp LED', 'Home Goods', 'Adjustable LED desk lamp with touch control', 39.99, 201, 'LightingCo', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (15, 'Backpack Travel', 'Accessories', 'Durable travel backpack with laptop compartment', 69.99, 188, 'TravelGear', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (16, 'Bluetooth Speaker', 'Electronics', 'Portable waterproof bluetooth speaker', 49.99, 245, 'AudioMax', NOW())

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (17, 'Yoga Mat Premium', 'Sports', 'Extra thick yoga mat with carrying strap', 34.99, 167, 'FitnessPro', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (18, 'Stainless Steel Water Bottle', 'Home Goods', 'Insulated water bottle keeps drinks cold for 24 hours', 24.99, 389, 'HydroBottle', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (19, 'Winter Jacket Mens', 'Clothing', 'Warm winter jacket with hood', N/A, 93, 'OutdoorWear', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (20, 'Electric Toothbrush', 'Personal Care', 'Rechargeable electric toothbrush with multiple modes', 59.99, 134, 'DentalCare', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (21, 'Chef's Knife Set', 'Kitchen', 'Professional knife set with wooden block', 149.99, 52, 'CulinaryTools', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (22, 'Monitor 27-inch 4K', 'Electronics', 'Ultra HD 4K monitor for professionals', 449.99, 41, 'DisplayTech', NOW())

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (23, 'Air Purifier Home', 'Appliances', 'HEPA filter air purifier for large rooms', 199.99, 68, 'CleanAir', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (24, 'Sunglasses Polarized', 'Accessories', 'UV protection polarized sunglasses', 89.99, 214, 'EyewearPlus', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (25, 'Fitness Tracker Watch', 'Electronics', 'Activity tracker with heart rate monitor', 79.99, 298, 'HealthTech', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (13, 'Blender Pro 1000W', 'Appliances', 'High-power blender for smoothies and more', 129.99, 87, 'KitchenPro', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (27, 'Bath Towel Set', 'Home Goods', 'Soft cotton bath towel set of 6 pieces', 44.99, 176, 'HomeLinens', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (28, 'Mechanical Keyboard RGB', 'Electronics', 'Gaming mechanical keyboard with

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (29, 'Protein Powder Vanilla', 'Health', 'Whey protein powder 5lb container', 49.99, 203, 'NutriFit', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (30, 'Dining Table Set', 'Furniture', 'Modern dining table with 4 chairs', 599.99, 23, 'HomeFurnish', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (31, 'Women's Running Shoes', 'Footwear', 'Lightweight running shoes for women', 94.99, 128, 'SportFit', NOW())

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (32, 'Electric Kettle', 'Appliances', 'Fast-boiling electric kettle with auto shut-off', 34.99, 267, 'KitchenMaster', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (33, 'Men's Dress Shirt', 'Clothing', 'Formal dress shirt wrinkle-free fabric', 39.99, 187, 'FormalWear', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (34, 'Webcam HD 1080p', 'Electronics', 'High definition webcam for video calls', 69.99, 143, 'VideoTech', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (35, 'Throw Pillows Set', 'Home Goods', 'Decorative throw pillows set of 4', 29.99, 234, 'HomeDecor', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (36, 'Power Bank 20000mAh', 'Electronics', 'High capacity portable charger', '$45.99', 312, 'ChargeMax', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (37, 'Vacuum Cleaner Robot', 'Appliances', 'Smart robot vacuum with app control', 299.99, 56, 'CleanBot', NOW())

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (38, 'Camping Tent 4-Person', 'Outdoor', 'Weatherproof camping tent for 4 people', 159.99, 74, 'OutdoorGear', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (39, 'Cookbook Mediterranean', 'Books', 'Mediterranean cuisine cookbook with 200 recipes', 24.99, 145, 'CookBooks Inc', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (40, 'Baby Monitor Video', 'Baby Products', 'Video baby monitor with night vision', 119.99, 89, 'BabyCare', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (41, 'Guitar Acoustic', 'Musical Instruments', 'Beginner's acoustic guitar with case', 179.99, 43, 'MusicMakers', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (42, 'Tool Set 100-Piece', 'Tools', 'Complete tool set for home repairs', 89.99, 112, 'ToolMaster', NOW())

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (43, 'Smart Light Bulbs 4-Pack', 'Smart Home', 'WiFi enabled color changing smart bulbs', 54.99, 198, 'SmartHome', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (44, 'Dog Food Premium 30lb', 'Pet Supplies', 'High-quality dog food grain-free formula', 64.99, 156, 'PetNutrition', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (45, 'Wall Clock Modern', 'Home Goods', 'Minimalist wall clock silent mechanism', 34.99, 223, 'TimeKeepers', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (46, 'Basketball Official Size', 'Sports', 'Official size basketball for indoor/outdoor', 29.99, 189, 'SportsPro', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (47, 'Printer All-in-One', 'Electronics', 'Wireless all-in-one printer scanner copier', 199.99, 64, 'PrintTech', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (48, 'Children's Bicycle 20-inch', 'Toys', 'Kids bicycle with training wheels', 149.99, 0, 'KidRide', NOW())

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (49, 'External Hard Drive 2TB', 'Electronics', 'Portable external hard drive USB 3.0', 79.99, 237, 'StorageMax', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (50, 'Women's Winter Boots', 'Footwear', 'Warm waterproof winter boots for women', 109.99, 94, 'FootComfort', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (51, 'Microwave Oven 1000W', 'Appliances', 'Countertop microwave with digital controls', 129.99, 103, 'KitchenMaster', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (52, 'Canvas Wall Art Set', 'Home Decor', 'Abstract canvas wall art 3-piece set', 79.99, 145, 'ArtDecor', NOW())

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (53, 'Gaming Chair Ergonomic', 'Furniture', 'Professional gaming chair with adjustable armrests', 249.99, 58, 'GamerSeats', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (54, 'Slow Cooker 6-Quart', 'Appliances', 'Programmable slow cooker with timer', 49.99, 178, 'KitchenPro', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (55, 'Men's Leather Belt', 'Accessories', 'Genuine leather belt with classic buckle', 34.99, 265, 'LeatherCo', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (56, 'Desk Organizer Set', 'Office Supplies', 'Complete desk organization set mesh metal', 24.99, 312, 'OfficeMax', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (57, 'Electric Shaver Men's', 'Personal Care', 'Rechargeable men's electric shaver waterproof', '$89.99', 127, 'GroomTech', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (58, 'Memory Foam Pillow', 'Bedding', 'Contour memory foam pillow for neck support', 39.99, 289, 'SleepWell', NOW());

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (59, 'Smart Thermostat WiFi', 'Smart Home', 'Programmable WiFi thermostat with app', 179.99, 82, 'SmartHome', NOW())

INSERT INTO products (product_id, product_name, category, description, price, stock_quantity, manufacturer, created_at) VALUES (60, 'Cookbook Italian Classics', 'Books', 'Authentic Italian

/* End of dump - file truncated */