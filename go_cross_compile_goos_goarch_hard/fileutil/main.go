package main

import (
	"fmt"
	"os"
	
	"fileutil/fileutil"
)

func main() {
	fmt.Println("Processing files...")
	
	// Get current working directory
	dir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting working directory: %v\n", err)
		os.Exit(1)
	}
	
	// Process files in the current directory
	err = fileutil.ProcessFiles(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error processing files: %v\n", err)
		os.Exit(1)
	}
	
	fmt.Println("File processing completed successfully")
}