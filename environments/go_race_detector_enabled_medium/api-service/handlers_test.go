package main

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

func TestConcurrentIncrements(t *testing.T) {
	var wg sync.WaitGroup
	numGoroutines := 75

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			req := httptest.NewRequest("POST", "/increment", nil)
			rr := httptest.NewRecorder()
			
			incrementHandler(rr, req)
			
			if status := rr.Code; status != http.StatusOK {
				t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
			}
		}()
	}
	
	wg.Wait()
	time.Sleep(10 * time.Millisecond)
}

func TestConcurrentMixedOperations(t *testing.T) {
	var wg sync.WaitGroup
	operations := 30

	for i := 0; i < operations; i++ {
		wg.Add(3)
		
		go func() {
			defer wg.Done()
			req := httptest.NewRequest("POST", "/increment", nil)
			rr := httptest.NewRecorder()
			incrementHandler(rr, req)
			
			if status := rr.Code; status != http.StatusOK {
				t.Errorf("increment handler returned wrong status code: got %v want %v", status, http.StatusOK)
			}
		}()
		
		go func() {
			defer wg.Done()
			req := httptest.NewRequest("POST", "/decrement", nil)
			rr := httptest.NewRecorder()
			decrementHandler(rr, req)
			
			if status := rr.Code; status != http.StatusOK {
				t.Errorf("decrement handler returned wrong status code: got %v want %v", status, http.StatusOK)
			}
		}()
		
		go func() {
			defer wg.Done()
			req := httptest.NewRequest("GET", "/status", nil)
			rr := httptest.NewRecorder()
			statusHandler(rr, req)
			
			if status := rr.Code; status != http.StatusOK {
				t.Errorf("status handler returned wrong status code: got %v want %v", status, http.StatusOK)
			}
		}()
	}
	
	wg.Wait()
	time.Sleep(10 * time.Millisecond)
}

func TestSequentialOperations(t *testing.T) {
	req := httptest.NewRequest("POST", "/increment", nil)
	rr := httptest.NewRecorder()
	incrementHandler(rr, req)
	
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}
	
	req = httptest.NewRequest("GET", "/status", nil)
	rr = httptest.NewRecorder()
	statusHandler(rr, req)
	
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v", status, http.StatusOK)
	}
}