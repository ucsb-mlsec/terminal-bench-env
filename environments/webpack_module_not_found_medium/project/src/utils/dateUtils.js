// Date utility functions for the application

/**
 * Format a date object to a readable string
 * @param {Date} date - The date to format
 * @returns {string} Formatted date string
 */
export function formatDate(date) {
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
}

/**
 * Get the current date
 * @returns {Date} Current date object
 */
export function getCurrentDate() {
    return new Date();
}