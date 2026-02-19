// src/string-utils.js
// String manipulation utilities

/**
 * Capitalizes the first letter of a string
 * @param {string} str - Input string
 * @returns {string} Capitalized string
 */
export function capitalize(str) {
  if (!str || typeof str !== 'string') return '';
  // TODO: This only works for first character, needs to handle empty strings better
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
}

/**
 * Reverses a string
 * @param {string} str - Input string
 * @returns {string} Reversed string
 */
export function reverse(str) {
  // TODO: Implement string reversal
  // Should handle empty strings and non-string inputs
  return '';
}

/**
 * Truncates a string to specified length
 * @param {string} str - Input string
 * @param {number} length - Maximum length
 * @returns {string} Truncated string with '...' if needed
 */
export function truncate(str, length) {
  if (!str || typeof str !== 'string') return '';
  // BUG: This doesn't account for the '...' in the length calculation
  if (str.length > length) {
    return str.slice(0, length) + '...';
  }
  return str;
}

/**
 * Converts string to camelCase
 * @param {string} str - Input string (with - or _ separators)
 * @returns {string} camelCase string
 */
export function toCamelCase(str) {
  if (!str || typeof str !== 'string') return '';
  // TODO: Incomplete implementation
  // Should handle both 'hello-world' and 'hello_world' formats
  // Currently only partially working
  const words = str.split('-');
  // Missing: logic to capitalize subsequent words and handle underscores
  return words[0];
}