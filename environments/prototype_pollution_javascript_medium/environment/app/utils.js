/**
 * Utility module for object operations
 * WARNING: This implementation is VULNERABLE to prototype pollution
 */

/**
 * Check if a value is an object (not null, not array)
 * @param {*} obj - Value to check
 * @returns {boolean} - True if value is a plain object
 */
function isObject(obj) {
  return obj !== null && typeof obj === 'object' && !Array.isArray(obj);
}

/**
 * Deep merge source object into target object
 * VULNERABLE: Does not check for dangerous keys like __proto__, constructor, prototype
 * @param {Object} target - Target object to merge into
 * @param {Object} source - Source object to merge from
 * @returns {Object} - Merged target object
 */
function deepMerge(target, source) {
  // Simple merge without any security checks
  for (let key in source) {
    if (source.hasOwnProperty(key)) {
      const sourceValue = source[key];
      const targetValue = target[key];
      
      // If both values are objects, recursively merge
      if (isObject(targetValue) && isObject(sourceValue)) {
        deepMerge(targetValue, sourceValue);
      } else if (Array.isArray(sourceValue)) {
        // Replace arrays
        target[key] = sourceValue.slice();
      } else if (isObject(sourceValue)) {
        // Create new object and merge
        target[key] = {};
        deepMerge(target[key], sourceValue);
      } else {
        // Simple assignment - this allows pollution through __proto__, constructor, etc.
        target[key] = sourceValue;
      }
    }
  }
  
  return target;
}

/**
 * Clone an object deeply
 * @param {Object} obj - Object to clone
 * @returns {Object} - Deep cloned object
 */
function deepClone(obj) {
  if (!isObject(obj) && !Array.isArray(obj)) {
    return obj;
  }
  
  if (Array.isArray(obj)) {
    return obj.map(item => deepClone(item));
  }
  
  const cloned = {};
  for (let key in obj) {
    if (obj.hasOwnProperty(key)) {
      cloned[key] = deepClone(obj[key]);
    }
  }
  
  return cloned;
}

module.exports = {
  isObject,
  deepMerge,
  deepClone
};