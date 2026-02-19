package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

var counter int
var requestMap map[string]int

func init() {
	requestMap = make(map[string]int)
}

func IncrementHandler(w http.ResponseWriter, r *http.Request) {
	// Read the counter
	currentValue := counter
	
	// Simulate some processing time
	time.Sleep(10 * time.Millisecond)
	
	// Write back the incremented value (race condition here!)
	counter = currentValue + 1
	
	// Also update request map without synchronization
	requestMap["increment"] = requestMap["increment"] + 1
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "incremented",
		"value":  counter,
	})
}

func DecrementHandler(w http.ResponseWriter, r *http.Request) {
	// Read the counter
	currentValue := counter
	
	// Simulate some processing time
	time.Sleep(10 * time.Millisecond)
	
	// Write back the decremented value (race condition here!)
	counter = currentValue - 1
	
	// Also update request map without synchronization
	requestMap["decrement"] = requestMap["decrement"] + 1
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "decremented",
		"value":  counter,
	})
}

func StatusHandler(w http.ResponseWriter, r *http.Request) {
	// Read counter and map without synchronization
	currentCounter := counter
	
	incrementCount := requestMap["increment"]
	decrementCount := requestMap["decrement"]
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"counter":         currentCounter,
		"increment_calls": incrementCount,
		"decrement_calls": decrementCount,
	})
}

func ResetHandler(w http.ResponseWriter, r *http.Request) {
	// Reset without synchronization (race condition!)
	counter = 0
	requestMap = make(map[string]int)
	
	fmt.Fprintf(w, "Reset complete")
}