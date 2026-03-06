package main

import (
	"github.com/acmecorp/logger"
)

func main() {
	log := logger.New()
	
	log.Info("Application started")
	log.Debug("Debug message")
	log.Warn("This is a warning message")
	log.Error("Sample error")
	
	log.Info("Performing application initialization")
	log.Debug("Loading configuration files")
	log.Info("Application ready to process requests")
}