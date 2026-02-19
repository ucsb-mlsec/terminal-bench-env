// Storage abstraction layer for the data processing service
// This interface defines the contract that all storage backends must implement

//go:build cloud

package main

// Storage defines the interface for all storage backend implementations
// across different deployment configurations (cloud, edge, and default)
type Storage interface {
	// Connect establishes a connection to the storage backend
	Connect() error
	
	// Store saves a key-value pair to the storage backend
	Store(key, value string) error
	
	// Retrieve fetches a value by key from the storage backend
	Retrieve(key string) (string, error)
}