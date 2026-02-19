const csv = require('csv-parser');
const fs = require('fs');
const math = require('mathjs');
const stats = require('simple-statistics');
const path = require('path');

// Store parsed data
const data = [];
const numericColumns = {};

console.log('CSV Processor - Scientific Measurements Statistics Tool');
console.log('======================================================\n');

// Read and parse CSV file
const inputFile = path.join(__dirname, 'input.csv');

fs.createReadStream(inputFile)
  .pipe(csv())
  .on('data', (row) => {
    data.push(row);
    
    // Identify numeric columns and collect values
    Object.keys(row).forEach(column => {
      const value = parseFloat(row[column]);
      if (!isNaN(value)) {
        if (!numericColumns[column]) {
          numericColumns[column] = [];
        }
        numericColumns[column].push(value);
      }
    });
  })
  .on('end', () => {
    console.log(`Processed ${data.length} rows from input.csv\n`);
    
    if (Object.keys(numericColumns).length === 0) {
      console.log('No numeric columns found in the dataset.');
      process.exit(0);
    }
    
    // Compute statistics for each numeric column
    console.log('Statistical Analysis Results:');
    console.log('-----------------------------\n');
    
    Object.keys(numericColumns).forEach(column => {
      const values = numericColumns[column];
      
      try {
        const mean = math.mean(values);
        const median = stats.median(values);
        const stdDev = stats.standardDeviation(values);
        const min = math.min(values);
        const max = math.max(values);
        const sum = math.sum(values);
        
        console.log(`Column: ${column}`);
        console.log(`  Count: ${values.length}`);
        console.log(`  Mean: ${mean.toFixed(4)}`);
        console.log(`  Median: ${median.toFixed(4)}`);
        console.log(`  Std Dev: ${stdDev.toFixed(4)}`);
        console.log(`  Min: ${min.toFixed(4)}`);
        console.log(`  Max: ${max.toFixed(4)}`);
        console.log(`  Sum: ${sum.toFixed(4)}`);
        console.log('');
      } catch (error) {
        console.error(`Error computing statistics for column ${column}:`, error.message);
      }
    });
    
    console.log('Analysis complete!');
    process.exit(0);
  })
  .on('error', (error) => {
    console.error('Error reading CSV file:', error.message);
    process.exit(1);
  });

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});