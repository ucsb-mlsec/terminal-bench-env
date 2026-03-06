// JavaScript Utility Library
// A comprehensive collection of utility functions for common operations

/**
 * Capitalizes the first letter of a string
 * @param {string} str - The input string
 * @return {string} The capitalized string
 * @export
 */
function capitalize(str) {
  if (typeof str !== 'string' || str.length === 0) {
    return str;
  }
  const firstCharacter = str.charAt(0);
  const remainingCharacters = str.slice(1);
  return firstCharacter.toUpperCase() + remainingCharacters.toLowerCase();
}

/**
 * Truncates a string to a specified length and adds ellipsis
 * @param {string} str - The input string
 * @param {number} maxLength - Maximum length before truncation
 * @return {string} The truncated string
 * @export
 */
function truncate(str, maxLength) {
  if (typeof str !== 'string') {
    return str;
  }
  if (str.length <= maxLength) {
    return str;
  }
  const truncatedString = str.substring(0, maxLength);
  const ellipsisCharacters = '...';
  return truncatedString + ellipsisCharacters;
}

/**
 * Converts a string to URL-friendly slug format
 * @param {string} str - The input string
 * @return {string} The slugified string
 * @export
 */
function slugify(str) {
  if (typeof str !== 'string') {
    return '';
  }
  const lowercaseString = str.toLowerCase();
  const trimmedString = lowercaseString.trim();
  const replacedString = trimmedString.replace(/[^\w\s-]/g, '');
  const finalSlug = replacedString.replace(/[\s_-]+/g, '-');
  const cleanedSlug = finalSlug.replace(/^-+|-+$/g, '');
  return cleanedSlug;
}

/**
 * Returns an array of unique values
 * @param {Array} arr - The input array
 * @return {Array} Array with unique values only
 * @export
 */
function unique(arr) {
  if (!Array.isArray(arr)) {
    return [];
  }
  const uniqueSet = new Set(arr);
  const uniqueArray = Array.from(uniqueSet);
  return uniqueArray;
}

/**
 * Flattens a nested array to a single level
 * @param {Array} arr - The nested array
 * @return {Array} The flattened array
 * @export
 */
function flatten(arr) {
  if (!Array.isArray(arr)) {
    return [];
  }
  const flattenedArray = arr.reduce((accumulator, currentValue) => {
    if (Array.isArray(currentValue)) {
      const recursiveFlattened = flatten(currentValue);
      return accumulator.concat(recursiveFlattened);
    } else {
      return accumulator.concat(currentValue);
    }
  }, []);
  return flattenedArray;
}

/**
 * Splits an array into chunks of specified size
 * @param {Array} arr - The input array
 * @param {number} size - The chunk size
 * @return {Array} Array of chunks
 * @export
 */
function chunk(arr, size) {
  if (!Array.isArray(arr) || size <= 0) {
    return [];
  }
  const chunkedArray = [];
  let currentIndex = 0;
  while (currentIndex < arr.length) {
    const currentChunk = arr.slice(currentIndex, currentIndex + size);
    chunkedArray.push(currentChunk);
    currentIndex = currentIndex + size;
  }
  return chunkedArray;
}

/**
 * Creates a deep clone of an object
 * @param {Object} obj - The object to clone
 * @return {Object} The cloned object
 * @export
 */
function deepClone(obj) {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }
  if (obj instanceof Date) {
    const dateCopy = new Date(obj.getTime());
    return dateCopy;
  }
  if (obj instanceof Array) {
    const arrayClone = [];
    for (let i = 0; i < obj.length; i++) {
      const clonedElement = deepClone(obj[i]);
      arrayClone.push(clonedElement);
    }
    return arrayClone;
  }
  if (obj instanceof Object) {
    const objectClone = {};
    const objectKeys = Object.keys(obj);
    for (let i = 0; i < objectKeys.length; i++) {
      const currentKey = objectKeys[i];
      const clonedValue = deepClone(obj[currentKey]);
      objectClone[currentKey] = clonedValue;
    }
    return objectClone;
  }
}

/**
 * Merges multiple objects into one
 * @param {...Object} objects - Objects to merge
 * @return {Object} The merged object
 * @export
 */
function merge(...objects) {
  const mergedObject = {};
  for (let i = 0; i < objects.length; i++) {
    const currentObject = objects[i];
    if (typeof currentObject === 'object' && currentObject !== null) {
      const objectKeys = Object.keys(currentObject);
      for (let j = 0; j < objectKeys.length; j++) {
        const currentKey = objectKeys[j];
        mergedObject[currentKey] = currentObject[currentKey];
      }
    }
  }
  return mergedObject;
}

/**
 * Picks specified properties from an object
 * @param {Object} obj - The source object
 * @param {Array<string>} keys - Keys to pick
 * @return {Object} Object with picked properties
 * @export
 */
function pick(obj, keys) {
  if (typeof obj !== 'object' || obj === null || !Array.isArray(keys)) {
    return {};
  }
  const pickedObject = {};
  for (let i = 0; i < keys.length; i++) {
    const currentKey = keys[i];
    if (obj.hasOwnProperty(currentKey)) {
      pickedObject[currentKey] = obj[currentKey];
    }
  }
  return pickedObject;
}

/**
 * Validates if a string is a valid email address
 * @param {string} email - The email string to validate
 * @return {boolean} True if valid email
 * @export
 */
function isEmail(email) {
  if (typeof email !== 'string') {
    return false;
  }
  const emailRegularExpression = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const validationResult = emailRegularExpression.test(email);
  return validationResult;
}

/**
 * Validates if a string is a valid URL
 * @param {string} url - The URL string to validate
 * @return {boolean} True if valid URL
 * @export
 */
function isURL(url) {
  if (typeof url !== 'string') {
    return false;
  }
  const urlRegularExpression = /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/;
  const validationResult = urlRegularExpression.test(url);
  return validationResult;
}

/**
 * Formats a date object to readable string
 * @param {Date} date - The date object
 * @param {string} format - Format string (default: 'YYYY-MM-DD')
 * @return {string} Formatted date string
 * @export
 */
function formatDate(date, format = 'YYYY-MM-DD') {
  if (!(date instanceof Date) || isNaN(date)) {
    return '';
  }
  const yearValue = date.getFullYear();
  const monthValue = date.getMonth() + 1;
  const dayValue = date.getDate();
  const paddedMonth = String(monthValue).padStart(2, '0');
  const paddedDay = String(dayValue).padStart(2, '0');
  
  if (format === 'YYYY-MM-DD') {
    return `${yearValue}-${paddedMonth}-${paddedDay}`;
  } else if (format === 'MM/DD/YYYY') {
    return `${paddedMonth}/${paddedDay}/${yearValue}`;
  } else if (format === 'DD/MM/YYYY') {
    return `${paddedDay}/${paddedMonth}/${yearValue}`;
  }
  return `${yearValue}-${paddedMonth}-${paddedDay}`;
}

/**
 * Clamps a number between min and max values
 * @param {number} num - The number to clamp
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @return {number} The clamped number
 * @export
 */
function clamp(num, min, max) {
  if (typeof num !== 'number' || typeof min !== 'number' || typeof max !== 'number') {
    return num;
  }
  if (num < min) {
    return min;
  }
  if (num > max) {
    return max;
  }
  return num;
}

// Internal helper function - not exported (should be eliminated)
function internalHelper(value) {
  const processedValue = value * 2 + 10;
  return processedValue;
}

// Another internal helper - not exported (should be eliminated)
function unusedInternalFunction(data) {
  const transformedData = data.map(item => item * 3);
  return transformedData;
}

// Internal validation helper - not exported
function validateInput(input) {
  return input !== null && input !== undefined;
}

// Export all public functions
window['capitalize'] = capitalize;
window['truncate'] = truncate;
window['slugify'] = slugify;
window['unique'] = unique;
window['flatten'] = flatten;
window['chunk'] = chunk;
window['deepClone'] = deepClone;
window['merge'] = merge;
window['pick'] = pick;
window['isEmail'] = isEmail;
window['isURL'] = isURL;
window['formatDate'] = formatDate;
window['clamp'] = clamp;