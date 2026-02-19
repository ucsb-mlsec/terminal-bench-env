package main

import (
	"fmt"
	"sync"
	"time"
)

// Cache represents a simple in-memory cache
type Cache struct {
	data map[string]string
}

// NewCache creates a new cache instance
func NewCache() *Cache {
	return &Cache{
		data: make(map[string]string),
	}
}

// Set stores a key-value pair in the cache
func (c *Cache) Set(key, value string) {
	c.data[key] = value
}

// Get retrieves a value from the cache
func (c *Cache) Get(key string) (string, bool) {
	val, ok := c.data[key]
	return val, ok
}

// Delete removes a key from the cache
func (c *Cache) Delete(key string) {
	delete(c.data, key)
}

func main() {
	cache := NewCache()
	var wg sync.WaitGroup

	// Spawn multiple goroutines to access cache concurrently
	numGoroutines := 20
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			
			key := fmt.Sprintf("key-%d", id%5)
			value := fmt.Sprintf("value-%d", id)
			
			// Perform multiple operations
			cache.Set(key, value)
			time.Sleep(time.Millisecond * 10)
			
			_, _ = cache.Get(key)
			time.Sleep(time.Millisecond * 5)
			
			cache.Set(key, value+"-updated")
			_, _ = cache.Get(key)
			
			if id%3 == 0 {
				cache.Delete(key)
			}
		}(i)
	}

	wg.Wait()
	fmt.Println("Application completed successfully")
}