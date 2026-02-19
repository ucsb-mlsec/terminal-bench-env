package main

import (
	"strconv"
	"sync"
	"testing"
)

func TestCacheBasicOperations(t *testing.T) {
	c := NewCache()

	// Test setting a value
	err := c.Set("key1", "value1")
	if err != nil {
		t.Fatalf("Failed to set value: %v", err)
	}

	// Test getting the value back
	val, err := c.Get("key1")
	if err != nil {
		t.Fatalf("Failed to get value: %v", err)
	}
	if val != "value1" {
		t.Errorf("Expected 'value1', got '%s'", val)
	}

	// Test deleting the value
	err = c.Delete("key1")
	if err != nil {
		t.Fatalf("Failed to delete value: %v", err)
	}

	// Test that deleted key returns error
	_, err = c.Get("key1")
	if err == nil {
		t.Error("Expected error for deleted key, got nil")
	}
}

func TestCacheConcurrentAccess(t *testing.T) {
	c := NewCache()
	var wg sync.WaitGroup

	numGoroutines := 100

	// Spawn goroutines that set values
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			key := "key" + strconv.Itoa(id)
			value := "value" + strconv.Itoa(id)
			err := c.Set(key, value)
			if err != nil {
				t.Errorf("Failed to set key %s: %v", key, err)
			}
		}(i)
	}

	// Spawn goroutines that get values
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			key := "key" + strconv.Itoa(id%50)
			_, _ = c.Get(key)
		}(i)
	}

	// Spawn some goroutines that delete keys
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			key := "key" + strconv.Itoa(id)
			_ = c.Delete(key)
		}(i)
	}

	wg.Wait()
}

func TestCacheConcurrentReads(t *testing.T) {
	c := NewCache()

	// Pre-populate cache with 10 entries
	for i := 0; i < 10; i++ {
		key := "key" + strconv.Itoa(i)
		value := "value" + strconv.Itoa(i)
		err := c.Set(key, value)
		if err != nil {
			t.Fatalf("Failed to set key %s: %v", key, err)
		}
	}

	var wg sync.WaitGroup

	// Spawn 50 goroutines that all read the same keys concurrently
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				key := "key" + strconv.Itoa(j)
				val, err := c.Get(key)
				if err != nil {
					continue
				}
				expectedValue := "value" + strconv.Itoa(j)
				if val != expectedValue {
					t.Errorf("Expected '%s', got '%s'", expectedValue, val)
				}
			}
		}(i)
	}

	wg.Wait()
}