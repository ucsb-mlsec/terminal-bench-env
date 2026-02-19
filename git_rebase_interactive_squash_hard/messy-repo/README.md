# Microservices Project

A microservices architecture project with user authentication, payment gateway, and notification services.

## Features

- User Authentication
- Payment Gateway
- Notification Service

## Setup

To set up this project locally:

1. Clone the repository
2. Install dependencies for each service
3. Configure environment variables
4. Run the services using the provided scripts

Detailed setup instructions will be provided for each microservice in their respective directories.

## Architecture

This project follows a microservices architecture pattern where each service is independently deployable and scalable. The services communicate with each other through REST APIs and message queues. Each microservice is responsible for a specific business capability:

- **User Authentication Service**: Handles user registration, login, and token management
- **Payment Gateway Service**: Processes payment transactions and integrates with third-party payment providers
- **Notification Service**: Manages email, SMS, and push notifications across the platform

The services are designed to be loosely coupled, allowing for independent development, testing, and deployment cycles.