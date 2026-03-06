const axios = require('axios');
const fs = require('fs');
const { resetQueryCount, getQueryCount } = require('./database.js');

async function runTest() {
  try {
    // Read the test query from file
    const query = fs.readFileSync('./test-query.txt', 'utf8');
    
    // Reset the query counter
    resetQueryCount();
    
    // Record start time
    const startTime = Date.now();
    
    // Send the GraphQL query
    const response = await axios.post('http://localhost:4000/graphql', {
      query: query
    });
    
    // Record end time and calculate duration
    const endTime = Date.now();
    const duration = endTime - startTime;
    
    // Get the total number of queries executed
    const queryCount = getQueryCount();
    
    // Print results to console
    console.log('\n=== Performance Test Results ===');
    console.log(`Database Queries Executed: ${queryCount}`);
    console.log(`Execution Time: ${duration}ms`);
    
    // Check if response has data
    if (response.data && response.data.data && response.data.data.products) {
      console.log(`\nProducts Retrieved: ${response.data.data.products.length}`);
      console.log('\nSample Product:');
      console.log(JSON.stringify(response.data.data.products[0], null, 2));
    }
    
    // Check for errors
    if (response.data && response.data.errors) {
      console.error('\nGraphQL Errors:', response.data.errors);
    }
    
    // Write results to solution.json
    const solution = {
      query_count: queryCount,
      duration_ms: Math.floor(duration)
    };
    
    fs.writeFileSync('../solution.json', JSON.stringify(solution, null, 2));
    console.log('\n✓ Results saved to solution.json');
    
    // Success criteria check
    if (queryCount <= 2) {
      console.log('\n✓ SUCCESS: Query count meets target (≤ 2 queries)');
    } else {
      console.log(`\n✗ NEEDS OPTIMIZATION: Query count is ${queryCount} (target: ≤ 2)`);
    }
    
  } catch (error) {
    console.error('Error running test:', error.message);
    if (error.response) {
      console.error('Response data:', error.response.data);
    }
    process.exit(1);
  }
}

runTest();