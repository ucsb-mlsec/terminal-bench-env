/**
 * merge-utils.js
 * Utility functions for deep merging objects
 * Used throughout the application for configuration and data manipulation
 */

/**
 * Check if a value is a plain object
 * @param {*} obj - Value to check
 * @returns {boolean} True if the value is a plain object
 */
function isObject(obj) {
  return obj !== null && typeof obj === 'object' && !Array.isArray(obj);
}

/**
 * Deep merge source object into target object
 * Recursively merges all properties from source into target
 * @param {Object} target - The target object to merge into
 * @param {Object} source - The source object to merge from
 * @returns {Object} The merged target object
 */
function deepMerge(target, source) {
  // Ensure both arguments are objects
  if (!isObject(target) || !isObject(source)) {
    return target;
  }

  // Iterate through all properties in the source object
  for (let key in source) {
    // Check if property exists directly on source (not inherited)
    if (source.hasOwnProperty(key)) {
      const sourceValue = source[key];
      
      // If the value is an object, recurse into it
      if (isObject(sourceValue)) {
        // If target doesn't have this key or it's not an object, create new object
        if (!target[key] || !isObject(target[key])) {
          target[key] = {};
        }
        // Recursively merge nested objects
        deepMerge(target[key], sourceValue);
      } else {
        // For non-object values, directly assign to target
        target[key] = sourceValue;
      }
    }
  }

  return target;
}

/**
 * Merge multiple objects into a single object
 * @param {...Object} objects - Objects to merge
 * @returns {Object} New merged object
 */
function merge(...objects) {
  const result = {};
  for (const obj of objects) {
    deepMerge(result, obj);
  }
  return result;
}

module.exports = {
  deepMerge,
  merge,
  isObject
};