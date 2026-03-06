# Money Transfer API - Security Documentation

## Overview

This is a Flask-based REST API application designed to handle secure money transfers between accounts. The application implements CSRF (Cross-Site Request Forgery) protection to prevent unauthorized transfer requests from malicious websites.

## CSRF Protection

The API implements a robust CSRF protection mechanism to secure all money transfer operations:

- All POST requests to `/api/transfer` must include a valid CSRF token
- CSRF tokens are generated on a per-session basis and tied to the user's session
- Tokens must be obtained from the `/api/get_token` endpoint before initiating any transfer
- The server validates all tokens before processing transfer requests
- Invalid or missing tokens result in request rejection with a 403 Forbidden response

## API Endpoints

### GET /api/get_token

Returns a CSRF token for the current session. This token must be included in subsequent transfer requests.

**Response:**
```json
{
  "token": "generated_csrf_token_string"
}
```

### POST /api/transfer

Processes a money transfer request. Requires a valid CSRF token obtained from `/api/get_token`.

**Request Body:**
```json
{
  "amount": 1000,
  "recipient": "ACC123",
  "token": "valid_csrf_token"
}
```

## Request Format

Transfer requests must be sent as JSON with the following structure:

- `amount` (integer): The amount of money to transfer
- `recipient` (string): The recipient account identifier
- `token` (string): Valid CSRF token obtained from `/api/get_token`

## Security Guarantees

The implementation provides protection against CSRF attacks by:

- Requiring tokens for all state-changing operations
- Validating token authenticity on the server side
- Preventing replay attacks through token expiration
- Ensuring tokens cannot be predicted or forged by attackers

## Example Usage

### Obtaining a CSRF token:

```bash
curl -X GET http://localhost:8080/api/get_token \
  -c cookies.txt
```

### Performing a transfer:

```bash
curl -X POST http://localhost:8080/api/transfer \
  -b cookies.txt \
  -H "Content-Type: application/json" \
  -d '{"amount": 500, "recipient": "ACC456", "token": "obtained_token_here"}'