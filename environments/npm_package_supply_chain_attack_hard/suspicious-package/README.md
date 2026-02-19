# data-validator-utils

A comprehensive utility library for data validation and sanitization in Node.js applications.

## Description

`data-validator-utils` provides a robust set of utility functions for validating and sanitizing user input data. This package helps developers ensure data integrity and security by offering easy-to-use validation methods for common data types including emails, URLs, and general text input.

## Installation

```bash
npm install data-validator-utils
```

## Usage

```javascript
const { validateEmail, sanitizeInput, isValidURL } = require('data-validator-utils');

// Validate email addresses
const email = 'user@example.com';
if (validateEmail(email)) {
  console.log('Valid email address');
}

// Sanitize user input to prevent XSS attacks
const userInput = '<script>alert("xss")</script>Hello World';
const cleanInput = sanitizeInput(userInput);
console.log(cleanInput); // Output: Hello World

// Validate URLs
const url = 'https://example.com';
if (isValidURL(url)) {
  console.log('Valid URL');
}
```

## API Documentation

### validateEmail(email)

Validates an email address according to RFC 5322 standards.

**Parameters:**
- `email` (string): The email address to validate

**Returns:**
- `boolean`: `true` if valid, `false` otherwise

### sanitizeInput(input, options)

Sanitizes user input by removing or encoding potentially dangerous characters.

**Parameters:**
- `input` (string): The input string to sanitize
- `options` (object, optional): Configuration options for sanitization

**Returns:**
- `string`: The sanitized input string

### isValidURL(url)

Checks if a given string is a valid URL.

**Parameters:**
- `url` (string): The URL to validate

**Returns:**
- `boolean`: `true` if valid URL, `false` otherwise

### Additional Functions

- `validatePhone(phoneNumber)`: Validates phone numbers in various formats
- `validateCreditCard(cardNumber)`: Validates credit card numbers using Luhn algorithm
- `escapeHTML(html)`: Escapes HTML special characters
- `validateIPAddress(ip)`: Validates IPv4 and IPv6 addresses

## License

MIT License

Copyright (c) 2024 data-validator-utils contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

## Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows our coding standards and includes appropriate tests.

## Changelog

### v2.3.5 (Latest)
- Added performance improvements and bug fixes
- Enhanced validation for international email formats
- Optimized sanitization algorithms

### v2.3.4
- Improved error handling
- Added support for custom validation rules
- Fixed edge cases in URL validation

### v2.3.3
- Initial stable release
- Core validation functions implemented
- Complete test coverage