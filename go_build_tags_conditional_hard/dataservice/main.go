// +build cloud edge
//go:build cloud || edge

package main

import (
	"fmt"
	"log"
)

type Storage interface {
	Connect() error
	Store(key string, value []byte) error
	Retrieve(key string) ([]byte, error)
	Close() error
}

func main() {
	fmt.Println("Starting data service...")
	
	storage := NewStorageBackend()
	if storage == nil {
		log.Fatal("Failed to create storage backend")
	}
	
	err := storage.Connect()
	if err != nil {
		log.Fatalf("Failed to connect to storage: %v", err)
	}
	defer storage.Close()
	
	fmt.Println("Storage backend initialized successfully")
	
	// Test storage operations
	testData := []byte("test data")
	err = storage.Store("test-key", testData)
	if err != nil {
		log.Printf("Warning: Failed to store test data: %v", err)
	}
	
	data, err := storage.Retrieve("test-key")
	if err != nil {
		log.Printf("Warning: Failed to retrieve test data: %v", err)
	} else {
		fmt.Printf("Retrieved data: %s\n", string(data))
	}
	
	fmt.Println("Data service running...")
}