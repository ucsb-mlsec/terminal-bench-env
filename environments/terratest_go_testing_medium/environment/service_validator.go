package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
)

type ServiceResponse struct {
	Status      string `json:"status"`
	ConfigValid bool   `json:"config_valid"`
	Service     string `json:"service"`
}

func ValidateEndpoint(url string, expectedStatus int) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to make GET request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != expectedStatus {
		return fmt.Errorf("expected status code %d, got %d", expectedStatus, resp.StatusCode)
	}

	return nil
}

func ValidateJSONResponse(url string, expectedFields []string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to make GET request: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response body: %v", err)
	}

	var jsonData map[string]interface{}
	err = json.Unmarshal(body, &jsonData)
	if err != nil {
		return fmt.Errorf("failed to parse JSON response: %v", err)
	}

	for _, field := range expectedFields {
		if _, exists := jsonData[field]; !exists {
			return fmt.Errorf("expected field '%s' not found in response", field)
		}
	}

	return nil
}

func GetServiceResponse(url string) (*ServiceResponse, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to make GET request: %v", err)
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}

	var serviceResp ServiceResponse
	err = json.Unmarshal(body, &serviceResp)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON into ServiceResponse: %v", err)
	}

	return &serviceResp, nil
}

func CheckServiceHealth(url string) (bool, error) {
	serviceResp, err := GetServiceResponse(url)
	if err != nil {
		return false, err
	}

	if serviceResp.Status == "healthy" {
		return true, nil
	}

	return false, errors.New("service is not healthy")
}