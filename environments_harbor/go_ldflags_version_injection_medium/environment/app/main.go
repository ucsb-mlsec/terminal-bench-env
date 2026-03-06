package main

import "fmt"

var Version string
var BuildTime string
var GitCommit string

func main() {
	fmt.Printf("Version: %s\n", Version)
	fmt.Printf("Build Time: %s\n", BuildTime)
	fmt.Printf("Git Commit: %s\n", GitCommit)
}