// main.go
package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <number>\n", os.Args[0])
		os.Exit(1)
	}

	num, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: invalid number: %v\n", err)
		os.Exit(1)
	}

	result := Calculate(num)
	fmt.Println(result)
}