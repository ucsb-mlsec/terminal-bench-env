package main

import (
	"fmt"
	"log"
	"net/http"
	"sync"
)

// Global counter accessed by multiple handlers - intentionally unsafe
var requestCounter int
var activeUsers = make(map[string]int)

// Mutex that exists but is not used (simulating forgotten synchronization)
var mu sync.Mutex

func incrementHandler(w http.ResponseWriter, r *http.Request) {
	// Race condition: reading and writing without lock
	requestCounter++
	userID := r.URL.Query().Get("user")
	if userID != "" {
		activeUsers[userID] = activeUsers[userID] + 1
	}
	fmt.Fprintf(w, "Counter: %d\n", requestCounter)
}

func decrementHandler(w http.ResponseWriter, r *http.Request) {
	// Race condition: reading and writing without lock
	requestCounter--
	userID := r.URL.Query().Get("user")
	if userID != "" {
		if count, exists := activeUsers[userID]; exists {
			activeUsers[userID] = count - 1
		}
	}
	fmt.Fprintf(w, "Counter: %d\n", requestCounter)
}

func statusHandler(w http.ResponseWriter, r *http.Request) {
	// Race condition: reading shared state without lock
	fmt.Fprintf(w, "Total requests: %d\n", requestCounter)
	fmt.Fprintf(w, "Active users: %d\n", len(activeUsers))
	for user, count := range activeUsers {
		fmt.Fprintf(w, "  %s: %d\n", user, count)
	}
}

func main() {
	http.HandleFunc("/increment", incrementHandler)
	http.HandleFunc("/decrement", decrementHandler)
	http.HandleFunc("/status", statusHandler)

	log.Println("Starting server on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}