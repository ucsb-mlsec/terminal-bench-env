package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"
)

var (
	showVersion = flag.Bool("version", false, "Display version information")
	showInfo    = flag.Bool("info", false, "Display system information")
	text        = flag.String("text", "", "Text to process (convert to uppercase)")
	repeat      = flag.Int("repeat", 1, "Number of times to repeat the text")
)

const version = "1.0.0"

func main() {
	flag.Parse()

	if *showVersion {
		displayVersion()
		return
	}

	if *showInfo {
		displaySystemInfo()
		return
	}

	if *text != "" {
		processText(*text, *repeat)
		return
	}

	// Default behavior: show help
	fmt.Println("CLI Tool - A simple command-line utility")
	fmt.Println("\nUsage:")
	flag.PrintDefaults()
	fmt.Println("\nExamples:")
	fmt.Println("  app --version")
	fmt.Println("  app --info")
	fmt.Println("  app --text \"hello world\" --repeat 3")
}

func displayVersion() {
	fmt.Printf("CLI Tool version %s\n", version)
	fmt.Printf("Built with Go %s\n", runtime.Version())
	fmt.Printf("Platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)
}

func displaySystemInfo() {
	fmt.Println("System Information:")
	fmt.Println("===================")
	fmt.Printf("Operating System: %s\n", runtime.GOOS)
	fmt.Printf("Architecture: %s\n", runtime.GOARCH)
	fmt.Printf("Go Version: %s\n", runtime.Version())
	fmt.Printf("CPUs: %d\n", runtime.NumCPU())
	fmt.Printf("Goroutines: %d\n", runtime.NumGoroutine())
	fmt.Printf("Compiler: %s\n", runtime.Compiler)
	fmt.Printf("Current Time: %s\n", time.Now().Format(time.RFC3339))
	
	hostname, err := os.Hostname()
	if err == nil {
		fmt.Printf("Hostname: %s\n", hostname)
	}
}

func processText(input string, times int) {
	if times < 1 {
		times = 1
	}
	
	processed := strings.ToUpper(input)
	fmt.Println("Processed Text:")
	fmt.Println("===============")
	
	for i := 0; i < times; i++ {
		fmt.Printf("%d: %s\n", i+1, processed)
	}
	
	fmt.Printf("\nOriginal length: %d characters\n", len(input))
	fmt.Printf("Word count: %d\n", len(strings.Fields(input)))
}