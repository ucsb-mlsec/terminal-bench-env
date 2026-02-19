package main

import (
	"fmt"
	"runtime"

	"app/processor"
)

// Main application entry point
// This application demonstrates conditional compilation with both pure Go
// and optional native code components for data processing operations.
// It can be built in multiple configurations depending on deployment needs.
func main() {
	fmt.Println("=== Multi-Configuration Go Application ===")
	fmt.Printf("Runtime: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("Go Version: %s\n", runtime.Version())
	fmt.Printf("CPUs: %d\n", runtime.NumCPU())
	fmt.Println()

	// Process some sample data using the processor package
	// The processor may use native code or pure Go depending on build configuration
	data := []byte("Hello from the application!")
	
	fmt.Println("Processing data...")
	result := processor.Process(data)
	
	fmt.Printf("Result: %s\n", result)
	fmt.Printf("Implementation: %s\n", processor.GetImplementation())
	
	fmt.Println()
	fmt.Println("Application completed successfully!")
}