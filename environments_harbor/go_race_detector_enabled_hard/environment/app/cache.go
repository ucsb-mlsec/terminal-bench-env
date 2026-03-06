package main

import (
	"errors"
)

// Cache is a simple in-memory cache implementation
type Cache struct {
	data map[string]interface{}
}

// NewCache creates and returns a new Cache instance
func NewCache() *Cache {
	return &Cache{
		data: make(map[string]interface{}),
	}
}

// Set stores a key-value pair in the cache
func (c *Cache) Set(key string, value interface{}) {
	c.data[key] = value
}

// Get retrieves a value from the cache by key
// Returns an error if the key is not found
func (c *Cache) Get(key string) (interface{}, error) {
	value, exists := c.data[key]
	if !exists {
		return nil, errors.New("key not found")
	}
	return value, nil
}

// Delete removes a key-value pair from the cache
func (c *Cache) Delete(key string) {
	delete(c.data, key)
}

// Clear removes all entries from the cache
func (c *Cache) Clear() {
	c.data = make(map[string]interface{})
}

// Len returns the number of items in the cache
func (c *Cache) Len() int {
	return len(c.data)
}