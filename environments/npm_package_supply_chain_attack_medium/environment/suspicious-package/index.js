/**
 * Data Formatter Utils
 * A collection of utility functions for formatting common data types
 * @module data-formatter-utils
 * @version 2.1.3
 */

/**
 * Formats a date object into a readable string format
 * @param {Date} date - The date object to format
 * @param {string} format - Format pattern (YYYY-MM-DD, MM/DD/YYYY, etc.)
 * @returns {string} Formatted date string
 */
function formatDate(date, format = 'YYYY-MM-DD') {
  if (!(date instanceof Date) || isNaN(date)) {
    throw new Error('Invalid date provided');
  }
  
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  
  return format
    .replace('YYYY', year)
    .replace('MM', month)
    .replace('DD', day);
}

/**
 * Formats a number as currency with proper symbols and separators
 * @param {number} amount - The amount to format
 * @param {string} currency - Currency code (USD, EUR, GBP, etc.)
 * @param {string} locale - Locale string for formatting rules
 * @returns {string} Formatted currency string
 */
function formatCurrency(amount, currency = 'USD', locale = 'en-US') {
  if (typeof amount !== 'number' || isNaN(amount)) {
    throw new Error('Invalid amount provided');
  }
  
  const formatter = new Intl.NumberFormat(locale, {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
  
  return formatter.format(amount);
}

/**
 * Formats a string by applying various transformations
 * @param {string} str - The string to format
 * @param {Object} options - Formatting options
 * @returns {string} Formatted string
 */
function formatString(str, options = {}) {
  if (typeof str !== 'string') {
    throw new Error('Invalid string provided');
  }
  
  let result = str;
  
  if (options.trim) result = result.trim();
  if (options.uppercase) result = result.toUpperCase();
  if (options.lowercase) result = result.toLowerCase();
  if (options.capitalize) {
    result = result.charAt(0).toUpperCase() + result.slice(1).toLowerCase();
  }
  
  return result;
}

module.exports = {
  formatDate,
  formatCurrency,
  formatString
};