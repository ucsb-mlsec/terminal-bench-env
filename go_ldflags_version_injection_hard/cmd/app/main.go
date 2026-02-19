package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/example/repo/pkg/version"
)

func main() {
	var showVersion bool
	flag.BoolVar(&showVersion, "version", false, "Display version information")
	flag.Parse()

	if showVersion {
		version.Print()
		os.Exit(0)
	}

	fmt.Println("Application running...")
}