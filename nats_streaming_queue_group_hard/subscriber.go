package main

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	stan "github.com/nats-io/stan.go"
)

// Buggy subscriber implementation - contains 5 bugs that cause duplicate message processing
func main() {
	// Bug 4: Wrong cluster ID - using incorrect cluster ID
	clusterID := "test-cluster"
	clientID := fmt.Sprintf("subscriber-%d", time.Now().Unix())
	
	// Connect to NATS Streaming server
	sc, err := stan.Connect(
		clusterID,
		clientID,
		stan.NatsURL("nats://localhost:4222"),
	)
	if err != nil {
		log.Fatalf("Failed to connect to NATS Streaming: %v", err)
	}
	defer sc.Close()

	log.Printf("Connected to NATS Streaming as client: %s", clientID)

	subject := "orders"
	
	// Message handler function
	msgHandler := func(msg *stan.Msg) {
		log.Printf("Received message: %s (sequence: %d)", string(msg.Data), msg.Sequence)
		
		// Process the message
		processOrder(msg.Data)
		
		// Bug 3: Missing message acknowledgment - msg.Ack() not called
		log.Printf("Processed message sequence: %d", msg.Sequence)
	}

	// Bug 1: Missing queue group name - empty string causes all instances to receive messages
	queueGroup := ""
	
	// Subscribe to the subject with buggy options
	// Bug 2: Wrong subscription option - SetManualAckMode(false) means auto-ack
	// Bug 5: StartWithLastReceived() is wrong for queue groups - causes redelivery issues
	sub, err := sc.QueueSubscribe(
		subject,
		queueGroup,
		msgHandler,
		stan.DurableName("order-processor"),
		stan.SetManualAckMode(false),
		stan.StartWithLastReceived(),
		stan.AckWait(30*time.Second),
	)
	if err != nil {
		log.Fatalf("Failed to subscribe: %v", err)
	}
	defer sub.Unsubscribe()

	log.Printf("Subscribed to subject '%s' with queue group '%s'", subject, queueGroup)
	log.Println("Waiting for messages... (Press Ctrl+C to exit)")

	// Setup signal handling to gracefully shutdown
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, os.Interrupt, syscall.SIGTERM)
	
	// Block until signal received
	<-signalChan
	
	log.Println("Shutting down subscriber...")
}

// processOrder simulates order processing
func processOrder(data []byte) {
	// Simulate processing time
	time.Sleep(100 * time.Millisecond)
	
	// In production, this would:
	// - Parse the order data
	// - Validate the order
	// - Store in database
	// - Trigger fulfillment workflow
	log.Printf("Processing order data: %s", string(data))
}