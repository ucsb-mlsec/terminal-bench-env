/**
 * API Client Module (BROKEN VERSION)
 * This implementation has timeout issues and will hang indefinitely on slow requests
 */

class ApiClient {
  /**
   * Creates a new API client instance
   * @param {Object} config - Configuration options
   * @param {number} config.timeout - Default timeout in milliseconds (NOT IMPLEMENTED)
   */
  constructor(config = {}) {
    this.config = config;
    this.activeRequests = new Set();
  }

  /**
   * Makes an HTTP request to the specified URL
   * BROKEN: This method does not implement timeout handling
   * Requests will hang indefinitely if the server doesn't respond
   * 
   * @param {string} url - The URL to request
   * @param {Object} options - Fetch options
   * @param {number} options.timeout - Request timeout in milliseconds (IGNORED)
   * @returns {Promise<any>} Response data
   */
  async makeRequest(url, options = {}) {
    try {
      // PROBLEM: No AbortController, no timeout mechanism
      // This will wait forever if the server doesn't respond
      const response = await fetch(url, {
        method: options.method || 'GET',
        headers: options.headers || {},
        body: options.body
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      // Basic error handling but no timeout-specific errors
      throw new Error(`Request failed: ${error.message}`);
    }
  }

  /**
   * Makes multiple concurrent requests
   * BROKEN: If any request hangs, the entire Promise.all will hang
   * 
   * @param {Array<{url: string, options: Object}>} requests - Array of request configurations
   * @returns {Promise<Array<any>>} Array of responses
   */
  async makeMultipleRequests(requests) {
    const promises = requests.map(req => 
      this.makeRequest(req.url, req.options)
    );
    
    // PROBLEM: No individual timeout handling, all requests wait indefinitely
    return await Promise.all(promises);
  }

  /**
   * Gets the number of active requests
   * BROKEN: activeRequests tracking is not actually used
   * 
   * @returns {number} Number of active requests
   */
  getActiveRequestCount() {
    return this.activeRequests.size;
  }
}

module.exports = ApiClient;