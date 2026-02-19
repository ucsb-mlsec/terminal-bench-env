//go:build edge
// +build edge

package main

import "fmt"

type EdgeStorage struct {
	LocalPath string
}

func NewStorageBackend() *EdgeStorage {
	return &EdgeStorage{
		LocalPath: "/var/local/data",
	}
}

func (e *EdgeStorage) Connect() error {
	fmt.Println("Connecting to edge storage...")
	return nil
}

// Missing Store() method - should implement:
// func (e *EdgeStorage) Store(key string, data []byte) error

// Missing Retrieve() method - should implement:
// func (e *EdgeStorage) Retrieve(key string) ([]byte, error)