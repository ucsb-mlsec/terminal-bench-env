const utils = require('./utils.js');

// In-memory store for user preferences
const preferencesStore = {};

/**
 * Update user preferences by merging new updates with existing preferences
 * VULNERABLE: Does not validate against prototype pollution keys
 * @param {string} userId - The user identifier
 * @param {object} updates - The preference updates to merge
 * @returns {object} The updated preferences
 */
function updatePreferences(userId, updates) {
  // Get existing preferences or create empty object
  const existingPreferences = preferencesStore[userId] || {};
  
  // Merge updates into existing preferences (VULNERABLE)
  const updatedPreferences = utils.deepMerge(existingPreferences, updates);
  
  // Store the updated preferences
  preferencesStore[userId] = updatedPreferences;
  
  return updatedPreferences;
}

/**
 * Get user preferences
 * @param {string} userId - The user identifier
 * @returns {object} The user's preferences or empty object
 */
function getPreferences(userId) {
  return preferencesStore[userId] || {};
}

module.exports = {
  updatePreferences,
  getPreferences
};