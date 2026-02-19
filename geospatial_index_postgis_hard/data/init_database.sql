-- Legacy Location-Based Application Database Initialization
-- This creates a database with spatial data stored incorrectly as separate lat/lon columns
-- PostGIS extension is NOT enabled, no geometry columns exist, no spatial indexes

-- Create the poi_locations table with separate latitude and longitude columns
CREATE TABLE IF NOT EXISTS poi_locations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    latitude NUMERIC(10, 8) NOT NULL,
    longitude NUMERIC(11, 8) NOT NULL,
    category VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data with realistic New York City area coordinates
-- Latitude range: 40.70 to 40.88, Longitude range: -74.02 to -73.85

INSERT INTO poi_locations (name, latitude, longitude, category) VALUES
-- Restaurants
('The Golden Spoon', 40.75823456, -73.98567234, 'restaurant'),
('Mama Rosa Italian Kitchen', 40.76234567, -73.96789012, 'restaurant'),
('Tokyo Sushi Bar', 40.74567890, -73.99234567, 'restaurant'),
('Le Petit Bistro', 40.77345678, -73.95678901, 'restaurant'),
('Dragon Palace Chinese', 40.72456789, -73.98901234, 'restaurant'),
('The Burger Joint', 40.78901234, -73.97456789, 'restaurant'),
('Spice Garden Indian Cuisine', 40.71234567, -73.99876543, 'restaurant'),
('Mediterranean Delight', 40.79876543, -73.96234567, 'restaurant'),
('The Steakhouse', 40.75678901, -73.98123456, 'restaurant'),
('Green Leaf Vegetarian', 40.76789012, -73.97890123, 'restaurant'),
('Taco Fiesta', 40.73456789, -73.99567890, 'restaurant'),
('Pizza Paradise', 40.80123456, -73.95890123, 'restaurant'),
('The French Corner', 40.74890123, -73.98567890, 'restaurant'),
('Seoul Kitchen', 40.77234567, -73.96567890, 'restaurant'),
('The Breakfast Club', 40.72890123, -73.99234567, 'restaurant'),

-- Hotels
('Grand Plaza Hotel', 40.75456789, -73.98234567, 'hotel'),
('The Riverside Inn', 40.76567890, -73.97123456, 'hotel'),
('Skyline Suites', 40.78234567, -73.96345678, 'hotel'),
('Downtown Comfort Hotel', 40.73678901, -73.99012345, 'hotel'),
('The Metropolitan', 40.79456789, -73.95567890, 'hotel'),
('Harbor View Hotel', 40.71567890, -73.99789012, 'hotel'),
('City Center Lodge', 40.77890123, -73.97234567, 'hotel'),
('The Boutique Hotel', 40.74234567, -73.98789012, 'hotel'),
('Business Travel Inn', 40.80456789, -73.95123456, 'hotel'),
('The Historic Manor', 40.72234567, -73.99456789, 'hotel'),

-- Parks
('Central Green Park', 40.76890123, -73.97567890, 'park'),
('Riverside Park', 40.78567890, -73.96890123, 'park'),
('Memorial Gardens', 40.73890123, -73.98567890, 'park'),
('Lakeside Park', 40.75123456, -73.99123456, 'park'),
('Veterans Memorial Park', 40.79123456, -73.96123456, 'park'),
('Community Gardens', 40.72567890, -73.99678901, 'park'),
('Hilltop Park', 40.81234567, -73.94890123, 'park'),
('Waterfront Esplanade', 40.71890123, -74.00123456, 'park'),
('Sunset Park', 40.77567890, -73.97890123, 'park'),
('Oak Grove Park', 40.74567890, -73.98345678, 'park'),

-- Museums
('City Art Museum', 40.76123456, -73.98456789, 'museum'),
('Natural History Center', 40.77678901, -73.96789012, 'museum'),
('Contemporary Gallery', 40.74123456, -73.99234567, 'museum'),
('Science Discovery Museum', 40.79234567, -73.95890123, 'museum'),
('Historical Society Museum', 40.72678901, -73.99567890, 'museum'),
('Maritime Museum', 40.71345678, -74.00456789, 'museum'),
('Modern Art Space', 40.78345678, -73.96567890, 'museum'),
('Aviation Museum', 40.80890123, -73.94567890, 'museum'),
('Cultural Heritage Center', 40.73234567, -73.98890123, 'museum'),
('Childrens Museum', 40.75890123, -73.97678901, 'museum'),

-- Cafes
('Morning Brew Cafe', 40.75567890, -73.98678901, 'cafe'),
('The Coffee Bean', 40.76456789, -73.97345678, 'cafe'),
('Espresso Corner', 40.74678901, -73.99123456, 'cafe'),
('The Reading Room Cafe', 40.77123456, -73.96890123, 'cafe'),
('Artisan Coffee House', 40.73567890, -73.98901234, 'cafe'),
('The Daily Grind', 40.79678901, -73.95678901, 'cafe'),
('Sunshine Cafe', 40.72345678, -73.99890123, 'cafe'),
('The Cozy Cup', 40.80234567, -73.95345678, 'cafe'),
('Urban Roast', 40.75234567, -73.98234567, 'cafe'),
('The Tea House', 40.76890123, -73.97456789, 'cafe'),

-- Shopping
('Downtown Mall', 40.75890123, -73.98456789, 'shop'),
('Fashion Boutique', 40.76234567, -73.97678901, 'shop'),
('The Bookstore', 40.74456789, -73.99345678, 'shop'),
('Electronics Emporium', 40.77890123, -73.96234567, 'shop'),
('Vintage Finds', 40.73123456, -73.99012345, 'shop'),
('Sports Gear Store', 40.79567890, -73.95456789, 'shop'),
('Home Decor Gallery', 40.72890123, -73.99678901, 'shop'),
('The Gift Shop', 40.80567890, -73.94890123, 'shop'),
('Organic Market', 40.75456789, -73.98012345, 'shop'),
('Tech Central', 40.76678901, -73.97234567, 'shop'),

-- Entertainment
('Grand Cinema', 40.75678901, -73.98567890, 'entertainment'),
('The Theater District', 40.76789012, -73.97123456, 'entertainment'),
('Comedy Club Downtown', 40.74234567, -73.99456789, 'entertainment'),
('Live Music Venue', 40.78123456, -73.96456789, 'entertainment'),
('Bowling Alley', 40.73456789, -73.98678901, 'entertainment'),
('Arcade Paradise', 40.79890123, -73.95234567, 'entertainment'),
('Jazz Club', 40.72123456, -73.99789012, 'entertainment'),
('The Concert Hall', 40.80678901, -73.94678901, 'entertainment'),
('Family Fun Center', 40.75123456, -73.98890123, 'entertainment'),
('Escape Room Adventures', 40.76345678, -73.97567890, 'entertainment'),

-- Gyms and Fitness
('FitLife Gym', 40.75345678, -73.98234567, 'fitness'),
('Yoga Studio Zen', 40.76567890, -73.97456789, 'fitness'),
('CrossFit Downtown', 40.74567890, -73.99234567, 'fitness'),
('The Wellness Center', 40.77456789, -73.96678901, 'fitness'),
('Pilates Plus', 40.73678901, -73.98789012, 'fitness'),
('Marathon Training Gym', 40.79234567, -73.95678901, 'fitness'),
('Swim and Fitness Club', 40.72456789, -73.99567890, 'fitness'),
('Boxing Academy', 40.80345678, -73.95012345, 'fitness'),
('Dance Fitness Studio', 40.75678901, -73.98123456, 'fitness'),
('Rock Climbing Gym', 40.76123456, -73.97890123, 'fitness'),

-- Services
('City Library', 40.75234567, -73.98678901, 'library'),
('Community Center', 40.76890123, -73.97234567, 'community'),
('Medical Clinic', 40.74890123, -73.99123456, 'medical'),
('Dental Care Center', 40.77234567, -73.96567890, 'medical'),
('Pet Hospital', 40.73234567, -73.98901234, 'medical'),
('Public Library Branch', 40.79456789, -73.95890123, 'library'),
('Senior Center', 40.72678901, -73.99456789, 'community'),
('Youth Recreation Center', 40.80123456, -73.95234567, 'community'),
('Post Office', 40.75567890, -73.98345678, 'service'),
('Police Station', 40.76456789, -73.97678901, 'service'),

-- Bars and Nightlife
('The Irish Pub', 40.75890123, -73.98456789, 'bar'),
('Rooftop Lounge', 40.76678901, -73.97345678, 'bar'),
('The Wine Bar', 40.74678901, -73.99234567, 'bar'),
('Sports Bar & Grill', 40.77890123, -73.96456789, 'bar'),
('Cocktail Lounge', 40.73456789, -73.98890123, 'bar'),
('The Dive Bar', 40.79678901, -73.95567890, 'bar'),
('Whiskey Tavern', 40.72234567, -73.99678901, 'bar'),
('Nightclub District', 40.80456789, -73.94890123, 'bar'),
('The Beer Garden', 40.75456789, -73.98234567, 'bar'),
('Piano Bar', 40.76234567, -73.97678901, 'bar'),

-- Additional Mixed Locations
('Tourist Information Center', 40.75678901, -73.98567890, 'service'),
('Convention Center', 40.76789012, -73.97123456, 'venue'),
('Food Court Plaza', 40.74345678, -73.99345678, 'restaurant'),
('Farmers Market', 40.78234567, -73.96234567, 'market'),
('Antique Shop', 40.73567890, -73.98678901, 'shop'),
('Art Gallery', 40.79123456, -73.95678901, 'museum'),
('Bakery Delight', 40.72456789, -73.99567890, 'cafe'),
('Hardware Store', 40.80567890, -73.94678901, 'shop'),
('Photography Studio', 40.75234567, -73.98456789, 'service'),
('Music School', 40.76567890, -73.97345678, 'education'),
('Language Institute', 40.74567890, -73.99123456, 'education'),
('Car Rental Service', 40.77678901, -73.96567890, 'service'),
('Flower Shop', 40.73890123, -73.98789012, 'shop'),
('Jewelry Store', 40.79890123, -73.95234567, 'shop'),
('Spa and Wellness', 40.72678901, -73.99456789, 'wellness'),
('Barber Shop', 40.80234567, -73.95123456, 'service'),
('Dog Park', 40.75890123, -73.98234567, 'park'),
('Skate Park', 40.76234567, -73.97678901, 'park'),
('Observatory', 40.81456789, -73.94567890, 'museum'),
('Aquarium', 40.71234567, -74.00567890, 'museum');