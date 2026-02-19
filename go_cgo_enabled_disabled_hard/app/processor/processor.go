//go:build !cgo

package processor

import (
	"fmt"
	"strings"
)

// Process implements the pure Go version without native dependencies.
// This implementation is used when CGO is disabled or unavailable.
func Process(input string) string {
	// Pure Go implementation - no native code dependencies
	// Performs basic string processing operations
	processed := strings.ToUpper(input)
	processed = strings.ReplaceAll(processed, " ", "_")
	
	// Return result with identifier showing pure-go implementation was used
	result := fmt.Sprintf("pure-go-processed: %s", processed)
	
	return result
}