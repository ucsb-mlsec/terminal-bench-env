package main

import (
	"database/sql"
	"fmt"
	"log"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

func main() {
	// Open SQLite database connection
	db, err := sql.Open("sqlite3", "./sensors.db")
	if err != nil {
		log.Fatal("Failed to open database:", err)
	}
	defer db.Close()

	// Test the connection
	if err := db.Ping(); err != nil {
		log.Fatal("Failed to ping database:", err)
	}

	// Create sensor readings table
	createTableSQL := `CREATE TABLE IF NOT EXISTS sensor_readings (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		sensor_name TEXT NOT NULL,
		temperature REAL,
		humidity REAL,
		timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
	);`

	_, err = db.Exec(createTableSQL)
	if err != nil {
		log.Fatal("Failed to create table:", err)
	}

	// Insert sample sensor reading
	insertSQL := `INSERT INTO sensor_readings (sensor_name, temperature, humidity, timestamp)
		VALUES (?, ?, ?, ?)`

	_, err = db.Exec(insertSQL, "sensor_01", 23.5, 45.2, time.Now())
	if err != nil {
		log.Fatal("Failed to insert data:", err)
	}

	// Query and display sensor readings
	rows, err := db.Query("SELECT sensor_name, temperature, humidity, timestamp FROM sensor_readings ORDER BY timestamp DESC LIMIT 5")
	if err != nil {
		log.Fatal("Failed to query data:", err)
	}
	defer rows.Close()

	fmt.Println("Recent Sensor Readings:")
	fmt.Println("========================")
	for rows.Next() {
		var name string
		var temp, humid float64
		var timestamp time.Time
		if err := rows.Scan(&name, &temp, &humid, &timestamp); err != nil {
			log.Fatal("Failed to scan row:", err)
		}
		fmt.Printf("%s | Temp: %.1f°C | Humidity: %.1f%% | Time: %s\n",
			name, temp, humid, timestamp.Format("2006-01-02 15:04:05"))
	}

	fmt.Println("\nDatabase initialized successfully!")
}