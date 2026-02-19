// +build !cloud !edge

package main

import (
	"fmt"
	"sync"
)

type DefaultStorage struct {
	CloudEnabled bool
	EdgeEnabled  bool
	data         map[string]string
	mu           sync.RWMutex
}

func NewStorageBackend() (*DefaultStorage, error) {
	return &DefaultStorage{
		CloudEnabled: true,
		EdgeEnabled:  true,
		data:         make(map[string]string),
	}, nil
}

func (d *DefaultStorage) Connect() error {
	fmt.Println("Connecting to default storage (hybrid mode)...")
	return nil
}

func (d *DefaultStorage) Store(key, value string) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	if key == "" {
		return fmt.Errorf("key cannot be empty")
	}
	
	d.data[key] = value
	fmt.Printf("Stored key '%s' in default storage\n", key)
	return nil
}

func (d *DefaultStorage) Retrieve(key string) (string, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	
	value, exists := d.data[key]
	if !exists {
		return "", fmt.Errorf("key '%s' not found", key)
	}
	
	return value, nil
}

func (d *DefaultStorage) Close() error {
	fmt.Println("Closing default storage...")
	return nil
}