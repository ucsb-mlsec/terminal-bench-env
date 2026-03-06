// Array utility functions for common operations
// Part of the UtilLib utility library

/**
 * Split an array into chunks of specified size
 * @param {Array} array - The array to chunk
 * @param {number} size - The size of each chunk
 * @returns {Array} Array of chunks
 */
export function chunk(array, size) {
  // TODO: Implementation needed
  // Should handle edge cases: empty arrays, size <= 0, size > array.length
  // Should return array of subarrays with specified size
  return [];
}

/**
 * Flatten nested arrays one level deep
 * @param {Array} array - The array to flatten
 * @returns {Array} Flattened array
 */
export function flatten(array) {
  // Partial implementation - might not handle all edge cases
  if (!Array.isArray(array)) {
    return [];
  }
  
  let result = [];
  for (let item of array) {
    if (Array.isArray(item)) {
      result.push(...item);
    } else {
      result.push(item);
    }
  }
  return result;
  // BUG: Doesn't handle null or undefined in the array properly
}

/**
 * Return unique values from an array
 * @param {Array} array - The array to process
 * @returns {Array} Array with unique values
 */
export function unique(array) {
  if (!Array.isArray(array)) {
    return [];
  }
  
  // Current implementation uses Set but has issues with object comparison
  // Objects with same content are treated as different
  return [...new Set(array)];
  // BUG: [{ a: 1 }, { a: 1 }] returns two objects instead of one
  // Need to implement deep comparison for objects
}

/**
 * Sum all numbers in an array
 * @param {Array} array - Array of numbers
 * @returns {number} Sum of all numbers
 */
export function sum(array) {
  if (!Array.isArray(array) || array.length === 0) {
    return 0;
  }
  
  return array.reduce((acc, val) => {
    return acc + (typeof val === 'number' ? val : 0);
  }, 0);
}