-- PostgreSQL Database Dump
-- Sensor Measurements Data with Quality Issues

DROP TABLE IF EXISTS sensor_measurements;

CREATE TABLE sensor_measurements (
    sensor_id INTEGER NOT NULL,
    measurement_time TIMESTAMP,
    temperature NUMERIC(5,2),
    humidity NUMERIC(5,2),
    location VARCHAR(100)
);

-- Insert statements with intentional data quality issues
-- Issues include: NULL timestamps, duplicates, unsorted data

INSERT INTO sensor_measurements (sensor_id, measurement_time, temperature, humidity, location) VALUES
(101, '2024-03-15 14:30:00', 22.5, 65.0, 'Building A'),
(102, '2023-11-20 09:15:00', 18.3, 72.5, 'Building B'),
(101, '2024-01-10 08:00:00', 20.1, 68.0, 'Building A'),
(103, NULL, 25.0, 55.0, 'Warehouse 1'),
(104, '2024-02-28 16:45:00', 19.8, 70.5, 'Factory Floor'),
(101, '2023-12-05 11:30:00', 21.5, 63.5, 'Building A'),
(102, '2024-03-15 14:30:00', 23.0, 66.0, 'Building B'),
(103, '2024-01-22 13:20:00', 24.5, 58.0, 'Warehouse 1'),
(104, '2023-10-08 07:45:00', 17.2, 75.0, 'Factory Floor'),
(101, '2024-04-01 10:00:00', 26.0, 60.0, 'Building A');

INSERT INTO sensor_measurements (sensor_id, measurement_time, temperature, humidity, location) VALUES
(102, NULL, 22.0, 68.5, 'Building B'),
(103, '2023-09-15 15:30:00', 28.5, 52.0, 'Warehouse 1'),
(104, '2024-03-10 12:15:00', 21.0, 67.0, 'Factory Floor'),
(101, '2023-11-01 09:00:00', 19.5, 71.0, 'Building A'),
(102, '2024-01-18 14:45:00', 20.8, 69.5, 'Building B'),
(103, '2024-02-14 08:30:00', 22.3, 64.0, 'Warehouse 1'),
(104, NULL, 18.0, 73.5, 'Factory Floor'),
(101, '2024-01-10 08:00:00', 20.5, 67.5, 'Building A'),
(102, '2023-12-20 16:00:00', 17.8, 76.0, 'Building B'),
(103, '2024-04-05 11:15:00', 27.0, 56.5, 'Warehouse 1');

INSERT INTO sensor_measurements (sensor_id, measurement_time, temperature, humidity, location) VALUES
(104, '2023-11-28 10:30:00', 19.0, 72.0, 'Factory Floor'),
(101, NULL, 23.5, 62.0, 'Building A'),
(102, '2024-02-05 13:45:00', 21.8, 65.5, 'Building B'),
(103, '2023-10-30 07:00:00', 16.5, 77.5, 'Warehouse 1'),
(104, '2024-01-25 15:20:00', 22.8, 63.0, 'Factory Floor'),
(101, '2023-09-22 12:00:00', 29.0, 50.0, 'Building A'),
(102, '2024-03-20 09:30:00', 24.5, 61.0, 'Building B'),
(103, '2023-12-12 14:15:00', 18.5, 74.0, 'Warehouse 1'),
(104, '2024-04-08 08:45:00', 25.5, 59.0, 'Factory Floor'),
(101, '2024-02-18 16:30:00', 20.0, 70.0, 'Building A');

INSERT INTO sensor_measurements (sensor_id, measurement_time, temperature, humidity, location) VALUES
(102, NULL, 21.5, 68.0, 'Building B'),
(103, '2024-01-05 10:45:00', 19.8, 71.5, 'Warehouse 1'),
(104, '2023-09-28 13:00:00', 27.5, 54.0, 'Factory Floor'),
(101, '2024-03-25 11:20:00', 23.8, 64.5, 'Building A'),
(102, '2023-10-15 08:15:00', 17.0, 78.0, 'Building B'),
(103, '2024-02-28 16:45:00', 22.0, 66.5, 'Warehouse 1'),
(104, '2023-12-30 12:30:00', 18.8, 73.0, 'Factory Floor'),
(101, '2024-01-15 14:00:00', 21.3, 67.0, 'Building A'),
(102, '2023-11-08 09:45:00', 19.2, 72.5, 'Building B'),
(103, NULL, 24.0, 60.5, 'Warehouse 1');

INSERT INTO sensor_measurements (sensor_id, measurement_time, temperature, humidity, location) VALUES
(104, '2024-03-05 15:15:00', 26.5, 58.5, 'Factory Floor'),
(101, '2023-10-20 07:30:00', 16.8, 76.5, 'Building A'),
(102, '2024-01-28 13:00:00', 22.5, 65.0, 'Building B'),
(103, '2023-11-15 10:15:00', 20.3, 69.0, 'Warehouse 1'),
(104, '2024-02-10 08:00:00', 19.5, 71.5, 'Factory Floor'),
(101, '2024-03-15 14:30:00', 22.8, 64.8, 'Building A'),
(102, '2023-09-05 16:45:00', 30.0, 48.0, 'Building B'),
(103, '2024-04-12 12:45:00', 28.0, 55.5, 'Warehouse 1'),
(104, NULL, 21.0, 68.5, 'Factory Floor'),
(101, '2023-12-18 11:00:00', 18.0, 75.5, 'Building A');

INSERT INTO sensor_measurements (sensor_id, measurement_time, temperature, humidity, location) VALUES
(102, '2024-02-22 14:30:00', 23.5, 63.5, 'Building B'),
(103, '2023-10-05 09:20:00', 17.5, 77.0, 'Warehouse 1'),
(104, '2024-01-08 15:45:00', 20.8, 69.5, 'Factory Floor'),
(101, '2023-11-25 08:30:00', 19.0, 73.0, 'Building A'),
(102, NULL, 25.0, 60.0, 'Building B'),
(103, '2024-03-08 13:15:00', 24.8, 61.5, 'Warehouse 1'),
(104, '2023-09-12 10:00:00', 28.8, 51.5, 'Factory Floor'),
(101, '2024-04-15 16:00:00', 27.5, 57.0, 'Building A'),
(102, '2023-12-08 12:20:00', 18.5, 74.5, 'Building B'),
(103, '2024-01-20 07:45:00', 21.0, 68.0, 'Warehouse 1');

INSERT INTO sensor_measurements (sensor_id, measurement_time, temperature, humidity, location) VALUES
(104, '2024-02-25 14:10:00', 22.3, 66.0, 'Factory Floor'),
(101, NULL, 20.5, 70.5, 'Building A'),
(102, '2023-10-28 11:30:00', 16.0, 79.0, 'Building B'),
(103, '2024-03-18 09:00:00', 25.5, 59.5, 'Warehouse 1'),
(104, '2023-11-12 15:50:00', 19.8, 72.0, 'Factory Floor'),
(101, '2024-01-30 13:25:00', 21.8, 66.5, 'Building A'),
(102, '2023-09-18 08:40:00', 29.5, 49.0, 'Building B'),
(103, '2024-04-02 10:30:00', 26.8, 58.0, 'Warehouse 1'),
(104, '2023-12-22 16:15:00', 17.5, 76.0, 'Factory Floor'),
(101, '2024-02-08 12:00:00', 20.3, 69.0, 'Building A');

INSERT INTO sensor_measurements (sensor_id, measurement_time, temperature, humidity, location) VALUES
(102, '2024-03-28 14:45:00', 24.0, 62.5, 'Building B'),
(103, NULL, 22.5, 65.5, 'Warehouse 1'),
(104, '2023-10-18 09:15:00', 18.3, 75.5, 'Factory Floor'),
(101, '2024-01-05 10:45:00', 19.8, 71.0, 'Building A'),
(102, '2023-11-30 07:30:00', 17.8, 76.5, 'Building B'),
(103, '2024-02-15 15:00:00', 23.3, 64.0, 'Warehouse 1'),
(104, '2023-09-25 13:45:00', 30.5, 47.5, 'Factory Floor'),
(101, '2024-03-12 11:10:00', 25.0, 60.5, 'Building A'),
(102, '2023-12-28 08:20:00', 18.0, 75.0, 'Building B'),
(103, '2024-01-28 16:30:00', 21.5, 67.5, 'Warehouse 1');

INSERT INTO sensor_measurements (sensor_id, measurement_time, temperature, humidity, location) VALUES
(104, '2024-04-10 12:50:00', 27.8, 56.0, 'Factory Floor'),
(101, '2023-10-12 14:00:00', 16.5, 78.5, 'Building A'),
(102, '2024-02-12 10:25:00', 22.0, 66.5, 'Building B'),
(103, '2023-11-22 09:40:00', 20.0, 70.5, 'Warehouse 1'),
(104, NULL, 24.5, 61.5, 'Factory Floor'),
(101, '2024-01-22 13:20:00', 21.5, 67.5, 'Building A'),
(102, '2023-09-08 15:55:00', 31.0, 46.5, 'Building B'),
(103, '2024-03-22 08:15:00', 26.0, 59.0, 'Warehouse 1'),
(104, '2023-12-15 11:40:00', 18.8, 73.5, 'Factory Floor'),
(101, '2024-04-18 16:20:00', 28.5, 55.5, 'Building A');

INSERT INTO sensor_measurements (sensor_id, measurement_time, temperature, humidity, location) VALUES
(102, '2024-01-12 12:30:00', 20.5, 69.0, 'Building B'),
(103, '2023-10-25 10:05:00', 17.0, 77.5, 'Warehouse 1'),
(104, '2024-03-15 14:30:00', 24.3, 62.0, 'Factory Floor'),
(101, NULL, 23.0, 64.0, 'Building A'),
(102, '2023-11-18 08:50:00', 19.3, 72.0, 'Building B'),
(103, '2024-02-20 15:30:00', 22.8, 65.0, 'Warehouse 1'),
(104, '2023-09-30 13:10:00', 29.0, 50.5, 'Factory Floor'),
(101, '2024-01-25 09:35:00', 20.8, 68.5, 'Building A'),
(102, '2023-12-05 16:05:00', 17.3, 76.0, 'Building B'),
(103, '2024-04-08 11:50:00', 27.3, 57.5, 'Warehouse 1');

INSERT INTO sensor_measurements (sensor_id, measurement_time, temperature, humidity, location) VALUES
(104, '2024-02-02 14:20:00', 21.3, 67.0, 'Factory Floor'),
(101, '2023-10-08 07:45:00', 16.0, 79.5, 'Building A'),
(102, '2024-03-05 10:40:00', 23.8, 63.0, 'Building B'),
(103, NULL, 25.5, 59.5, 'Warehouse 1'),
(104, '2023-11-05 12:55:00', 19.5, 71.5, 'Factory Floor'),
(101, '2024-02-28 16:45:00', 22.5, 65.5, 'Building A'),
(102, '2023-09-20 14:25:00', 30.8, 48.5, 'Building B'),
(103, '2024-01-15 08:00:00', 20.0, 70.0, 'Warehouse 1'),
(104, '2023-12-10 11:15:00', 18.3, 74.5, 'Factory Floor'),
(101, '2024-04-05 15:40:00', 26.5, 58.5, 'Building A');

INSERT INTO sensor_measurements (sensor_id, measurement_time, temperature, humidity, location) VALUES
(102, '2024-01-08 13:05:00', 21.0, 68.5, 'Building B'),
(103, '2023-10-15 09:30:00', 17.8, 76.5, 'Warehouse 1'),
(104, '2024-03-20 15:15:00', 25.8, 60.0, 'Factory Floor'),
(101, '2023-11-10 10:20:00', 19.0, 73.5, 'Building A'),
(102, NULL, 24.0, 62.5, 'Building B'),
(103, '2024-02-05 12:45:00', 22.3, 66.0, 'Warehouse 1'),
(104, '2023-09-15 16:30:00', 29.5, 51.0, 'Factory Floor'),
(101, '2024-01-18 08:55:00', 20.5, 69.5, 'Building A'),
(102, '2023-12-25 14:10:00', 17.0, 77.0, 'Building B'),
(103, '2024-04-20 10:25:00', 28.8, 54.5, 'Warehouse 1');