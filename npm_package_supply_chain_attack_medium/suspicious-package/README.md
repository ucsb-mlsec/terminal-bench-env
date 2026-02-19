# data-formatter-utils

[![npm version](https://badge.fury.io/js/data-formatter-utils.svg)](https://badge.fury.io/js/data-formatter-utils)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/npm/dm/data-formatter-utils.svg)](https://www.npmjs.com/package/data-formatter-utils)

A lightweight and efficient utility library for formatting and transforming data in Node.js and browser environments.

## Description

`data-formatter-utils` provides a collection of commonly used data formatting functions to help developers quickly transform and standardize data across their applications. Whether you're working with dates, numbers, strings, or complex objects, this package offers simple, reliable utilities to handle your formatting needs.

## Installation

```bash
npm install data-formatter-utils
```

## Usage

### Basic Examples

```javascript
const { formatCurrency, formatDate, sanitizeString } = require('data-formatter-utils');

// Format currency values
const price = formatCurrency(1234.56, 'USD');
console.log(price); // Output: $1,234.56

// Format dates
const formattedDate = formatDate(new Date(), 'YYYY-MM-DD');
console.log(formattedDate); // Output: 2024-01-15

// Sanitize user input
const cleanString = sanitizeString('<script>alert("xss")</script>Hello');
console.log(cleanString); // Output: Hello
```

### Advanced Usage

```javascript
const { transformObject, normalizeData } = require('data-formatter-utils');

// Transform object keys
const data = { first_name: 'John', last_name: 'Doe' };
const camelCased = transformObject(data, 'camelCase');
console.log(camelCased); // Output: { firstName: 'John', lastName: 'Doe' }

// Normalize nested data structures
const normalized = normalizeData(complexData, { trimStrings: true, removeNull: true });
```

## API Documentation

### `formatCurrency(amount, currency, locale)`
Formats a numeric value as currency with proper symbols and separators.
- **amount** (Number): The numeric value to format
- **currency** (String): Currency code (USD, EUR, GBP, etc.)
- **locale** (String, optional): Locale for formatting (default: 'en-US')
- **Returns**: Formatted currency string

### `formatDate(date, format)`
Converts a date object into a formatted string.
- **date** (Date): The date to format
- **format** (String): Format pattern (YYYY-MM-DD, MM/DD/YYYY, etc.)
- **Returns**: Formatted date string

### `sanitizeString(input, options)`
Removes potentially dangerous characters and HTML from strings.
- **input** (String): The string to sanitize
- **options** (Object, optional): Sanitization options
- **Returns**: Cleaned string

### `transformObject(obj, caseType)`
Transforms object keys to specified case format.
- **obj** (Object): Object to transform
- **caseType** (String): Target case type ('camelCase', 'snake_case', 'kebab-case')
- **Returns**: Transformed object

### `normalizeData(data, options)`
Normalizes data structures by applying various cleanup operations.
- **data** (Any): Data to normalize
- **options** (Object): Normalization options
- **Returns**: Normalized data

## Features

- 🚀 Lightweight with zero dependencies
- 📦 Works in Node.js and browser environments
- 💪 Fully typed with TypeScript definitions
- ✅ Thoroughly tested with 100% code coverage
- 🔧 Flexible configuration options

## License

MIT © 2024 data-formatter-utils

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Author

Maintained by the data-formatter-utils team

## Support

If you encounter any issues or have questions, please file an issue on our [GitHub repository](https://github.com/data-utils/data-formatter-utils).