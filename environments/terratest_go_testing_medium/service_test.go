package main

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHealthCheckEndpoint(t *testing.T) {
	server := httptest.NewServer(HealthCheckHandler())
	defer server.Close()

	// WRONG: Using incorrect URL path
	resp, err := makeRequest(server.URL + "/wrong/path")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	// INCOMPLETE: Status code check is incomplete
	if resp.StatusCode {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}

	// MISSING: Should check for 'status' field in JSON response
	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)
}

func TestConfigValidation(t *testing.T) {
	server := httptest.NewServer(ConfigHandler())
	defer server.Close()

	resp, err := makeRequest(server.URL + "/config")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	// Only checks status code
	if resp.StatusCode != 200 {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}

	// MISSING: Should validate 'config_valid' field in response body
	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)
	
	// No validation of response body content
}

func TestMetricsEndpoint(t *testing.T) {
	server := httptest.NewServer(MetricsHandler())
	defer server.Close()

	resp, err := makeRequest(server.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	// INCORRECT: Expects 201 instead of 200
	if resp.StatusCode != 201 {
		t.Errorf("Expected status 201, got %d", resp.StatusCode)
	}

	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)

	// MISSING: Should validate 'requests' field exists
	// MISSING: Should validate 'errors' field exists
}

func TestServiceValidatorHealth(t *testing.T) {
	server := httptest.NewServer(HealthCheckHandler())
	defer server.Close()

	healthy := CheckServiceHealth(server.URL)

	// WRONG: Expects false instead of true
	if healthy != false {
		t.Errorf("Expected service health to be false, got %v", healthy)
	}
}

func TestValidateEndpointFunction(t *testing.T) {
	server := httptest.NewServer(HealthCheckHandler())
	defer server.Close()

	status := ValidateEndpoint(server.URL + "/health")

	// INCORRECT: Wrong expected status code
	expectedStatus := 404
	if status != expectedStatus {
		t.Errorf("Expected status %d, got %d", expectedStatus, status)
	}
}