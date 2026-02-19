package version

import "fmt"

// Variables that can be set at build time using ldflags
var (
	Version   string = "unknown"
	BuildDate string = "unknown"
	GitCommit string = "unknown"
)

// Print outputs the version information to stdout
func Print() {
	fmt.Printf("Version: %s\n", Version)
	fmt.Printf("BuildDate: %s\n", BuildDate)
	fmt.Printf("GitCommit: %s\n", GitCommit)
}