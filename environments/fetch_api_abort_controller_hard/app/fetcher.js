// Data Aggregation Service
// Fetches data from multiple endpoints with timeout protection

const fetchWithTimeout = async (url, timeout) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    console.log(`⏱️  Timeout triggered for ${url}`);
    controller.abort();
  }, timeout);

  try {
    const response = await fetch(url, { signal: controller.signal });
    // BUG: Missing clearTimeout(timeoutId) here!
    // The timer continues running even after successful fetch
    const data = await response.json();
    return data;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(`Request timeout after ${timeout}ms`);
    }
    throw error;
  }
};

const endpoints = [
  'http://localhost:3000/fast',
  'http://localhost:3000/slow',
  'http://localhost:3000/medium',
  'http://localhost:3000/veryslow'
];

const TIMEOUT_MS = 2000;

async function aggregateData() {
  console.log('🚀 Starting data aggregation service...\n');

  for (const endpoint of endpoints) {
    try {
      console.log(`📡 Fetching: ${endpoint}`);
      const data = await fetchWithTimeout(endpoint, TIMEOUT_MS);
      console.log(`✅ Success: ${endpoint}`, data);
    } catch (error) {
      console.log(`❌ Failed: ${endpoint} - ${error.message}`);
    }
    console.log('');
  }

  console.log('📊 Aggregation complete. Waiting for cleanup...');
}

aggregateData().catch(console.error);